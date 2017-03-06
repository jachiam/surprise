import gc
import numpy as np
import time
from rllab.algos.batch_polopt import BatchPolopt
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.sampler import parallel_sampler
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import theano
import theano.tensor as TT


from sandbox.jachiam.exploration_bonuses.no_bonus import NoBonus
from sandbox.jachiam.algos.plus.sampler_plus import BatchSamplerPlus

class PolicyGradientPlus(BatchPolopt, Serializable):
    """
    Policy Gradient base algorithm

    with optional data reuse and importance sampling,
    and exploration bonuses


    Can use this as a base class for VPG, ERWR, TNPG, TRPO, etc. by picking appropriate optimizers / arguments

    for VPG: use FirstOrderOptimizer
    for ERWR: set positive_adv to True
    for TNPG: use ConjugateGradient optimizer with max_backtracks=1
    for TRPO: use ConjugateGradient optimizer with max_backtracks>1
    for PPO: use PenaltyLBFGS optimzer

    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            all_paths=True,
            step_size=0.01,
            entropy_regularize=False,
            entropy_coeff=1e-4,
            entropy_coeff_decay=1,
            exploration_bonus=None,
            exploration_lambda=0.001,
            normalize_bonus=True,
            nonnegative_bonus_mean=False,
            batch_aggregate_n=1,
            batch_aggregate_coeff=0.5,
            relative_weights=False,
            importance_sampling=False,
            decision_weight_mode='pd',
            clip_IS_coeff_above=False,
            clip_IS_coeff_below=False,
            IS_coeff_upper_bound=5,
            IS_coeff_lower_bound=0,
            **kwargs):


        """
        :param batch_aggregate_n: use this many epochs of data (including current)
        :param batch_aggregate_coeff: used to make contribution of old data smaller. formula:

            If a batch has age j, it is weighted proportionally to

                                          batch_aggregate_coeff ** j,

            with these batch weights normalized.

            If you want every batch to have equal weight, set batch_aggregate_coeff = 1. 

        :param relative_weights: used to make contribution of old data invariant to how many
                                 more or fewer trajectories the old batch may have.
        :param importance_sampling: do or do not use importance sampling to reweight old data
        :param clip_IS_coeff: if true, clip the IS coefficients.
        :param IS_coeff_bound: if clip_IS_coeff, then IS coefficients are bounded by this value. 
        :param decision_weight_mode: either 'pd', per decision, or 'pt', per trajectory

        """

        Serializable.quick_init(self, locals())

        self.optimizer = optimizer
        self.all_paths = all_paths

        # npo
        self.step_size = step_size

        # entropy regularization
        self.entropy_regularize = entropy_regularize
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay

        # intrinsic motivation
        self.exploration_bonus = exploration_bonus
        self.exploration_lambda = exploration_lambda
        self.normalize_bonus = normalize_bonus
        self.nonnegative_bonus_mean = nonnegative_bonus_mean

        # importance sampling
        self.importance_sampling = importance_sampling
        self.decision_weight_mode = decision_weight_mode
        self.clip_IS_coeff_above = clip_IS_coeff_above
        self.clip_IS_coeff_below = clip_IS_coeff_below
        self.IS_coeff_upper_bound = IS_coeff_upper_bound
        self.IS_coeff_lower_bound = IS_coeff_lower_bound
        self.batch_aggregate_n = batch_aggregate_n
        self.batch_aggregate_coeff = batch_aggregate_coeff
        self.relative_weights = relative_weights

        super(PolicyGradientPlus, self).__init__(optimizer=optimizer, 
                                                 sampler_cls=BatchSamplerPlus,
                                                 **kwargs)
        

    @overrides
    def init_opt(self):
        self.start_time = time.time()
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        weights_var = ext.new_tensor(
            'weights',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        self.dist_info_vars_func = ext.compile_function(
                inputs=[obs_var] + state_info_vars_list,
                outputs=dist_info_vars,
                log_name="dist_info_vars"
            )

        # when we want to get D_KL( pi' || pi) for data that was sampled on 
        # some behavior policy pi_b, where pi' is the optimization variable
        # and pi is the policy of the previous iteration,
        # the dist_info in memory will correspond to pi_b and not pi. 
        # so we have to compute the dist_info for that data on pi, on the fly.

        ent = dist.entropy_sym(dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_ent = TT.sum(weights_var * ent * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
            mean_kl = TT.sum(weights_var * kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * weights_var * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_ent = TT.mean(weights_var * ent)
            max_kl = TT.max(kl)
            mean_kl = TT.mean(weights_var * kl)
            surr_loss = - TT.mean(lr * weights_var * advantage_var)

        if self.entropy_regularize:
            self.entropy_beta = theano.shared(self.entropy_coeff)
            surr_loss -= self.entropy_beta * mean_ent

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                         weights_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        f_kl = ext.compile_function(
            inputs=input_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )



    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log('optimizing policy...')
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages", "weights"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl, max_kl = self.opt_info['f_kl'](*all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        if self.entropy_regularize and not(self.entropy_coeff_decay == 1):
            current_entropy_coeff = self.entropy_beta.get_value() * self.entropy_coeff_decay
            self.entropy_beta.set_value(current_entropy_coeff)
            logger.record_tabular('EntropyCoeff', current_entropy_coeff)
        logger.record_tabular('Time',time.time() - self.start_time)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.log('optimization finished')


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            expl=self.exploration_bonus,
        )
