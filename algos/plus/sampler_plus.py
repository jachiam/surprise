import numpy as np
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.sampler.base import Sampler
from rllab.misc import ext
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util



def local_truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is almost equal to max_samples. This is done by
    removing extra paths at the end of the list. But here, we do NOT make the last path shorter.
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    return paths



class BatchSamplerPlus(Sampler):
    def __init__(self, algo, **kwargs):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.experience_replay = []
        self.env_interacts_memory = []
        self.env_interacts = 0
        self.total_env_interacts = 0
        self.mean_path_len = 0


    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )

        """log_likelihoods for importance sampling"""
        for path in paths:
            logli = self.algo.policy.distribution.log_likelihood(path["actions"],path["agent_infos"])
            path["log_likelihood"] = logli


        """keep data use per iteration approximately fixed"""
        if not(self.algo.all_paths):
            paths = local_truncate_paths(paths, self.algo.batch_size)

        """keep track of path length"""
        self.env_interacts = sum([len(path["rewards"]) for path in paths])
        self.total_env_interacts += self.env_interacts
        self.mean_path_len = float(self.env_interacts)/len(paths)

        """manage experience replay for old batch reuse"""
        self.experience_replay.append(paths)
        self.env_interacts_memory.append(self.env_interacts)
        if len(self.experience_replay) > self.algo.batch_aggregate_n:
            self.experience_replay.pop(0)
            self.env_interacts_memory.pop(0)

        return paths


    def process_samples(self, itr, paths):

        """
        we will ignore paths argument and only use experience replay.
        note: if algo.batch_aggregate_n = 1, then the experience replay will
        only contain the most recent batch, and so len(all_paths) == 1.
        """
        
        if self.algo.exploration_bonus:
            self.compute_exploration_bonuses_and_statistics()

        self.compute_epoch_weights()

        all_paths = []
        all_baselines = []
        all_returns = []
        self.IS_coeffs = [[] for paths in self.experience_replay]

        for paths, weight, age in zip(self.experience_replay,self.weights,self.age):
            b_paths, b_baselines, b_returns = self.process_single_batch(paths, weight, age)
            all_paths     += b_paths
            all_baselines += [b_baselines]
            all_returns   += [b_returns]

        samples_data = self.create_samples_dict(all_paths)

        """log all useful info"""
        self.record_statistics(itr, all_paths, all_baselines, all_returns)

        """update vf and exploration bonus model"""
        self.update_parametrized_models()

        return samples_data


    def compute_exploration_bonuses_and_statistics(self):

        for paths in self.experience_replay: 
            for path in paths:
                path["bonuses"] = self.algo.exploration_bonus.get_bonus(path)

        self.bonus_total =  sum([
                                sum([
                                    sum(path["bonuses"])
                                for path in paths])
                            for paths in self.experience_replay])

        self.bonus_mean = self.bonus_total / sum(self.env_interacts_memory)

        self.new_bonus_total = sum([sum(path["bonuses"]) for path in self.experience_replay[-1]])
        self.new_bonus_mean = self.new_bonus_total / self.env_interacts_memory[-1]

        self.bonus_baseline = self.algo.exploration_lambda * \
                              min(0,self.bonus_mean / max(1,np.abs(self.bonus_mean)))


    def compute_epoch_weights(self):
        """create weights, with highest weight on most recent batch"""
        self.raw_weights = np.array(
                        [self.algo.batch_aggregate_coeff**j for j in range(len(self.experience_replay))],
                        dtype='float'
                        )
        self.raw_weights /= sum(self.raw_weights)
        self.raw_weights = self.raw_weights[::-1]
        self.weights = self.raw_weights.copy()

        """reweight the weights by how many paths are in that batch """
        if self.algo.relative_weights:
            total_paths = sum([len(paths) for paths in self.experience_replay])
            for j in range(len(self.weights)):
                self.weights[j] *= total_paths / len(self.experience_replay[j])

        self.age = np.arange(len(self.experience_replay))[::-1]
        

    def process_single_batch(self, paths, weight, age):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]

            """exploration bonuses"""
            if self.algo.exploration_bonus:
                path["bonuses"] *= self.algo.exploration_lambda
                if self.algo.normalize_bonus:
                    path["bonuses"] /= max(1,np.abs(self.bonus_mean))
                if self.algo.nonnegative_bonus_mean:
                    path["bonuses"] -= self.bonus_baseline
                deltas += path["bonuses"]

            """recompute agent infos for old data"""
            """(necessary for correct reuse of old data)"""
            if age > 0:
                self.update_agent_infos(path)

            """importance sampling and batch aggregation"""
            path["weights"] = weight * np.ones_like(path["rewards"])
            if age > 0 and self.algo.importance_sampling:
                self.compute_and_apply_importance_weights(path,age)            

            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        return paths, baselines, returns


    def update_agent_infos(self,path):
        """
        this updates the agent dist infos (i.e, mean & variance of Gaussian policy dist)
        so that it can compute the probability of taking these actions on the most recent
        policy is.
        meanwhile, the log likelihood of taking the actions on the original behavior policy
        can still be found in path["log_likelihood"].
        """
        state_info_list = [path["agent_infos"][k] for k in self.algo.policy.state_info_keys]
        input_list = tuple([path["observations"]] + state_info_list)
        cur_dist_info = self.algo.dist_info_vars_func(*input_list)
        for k in self.algo.policy.distribution.dist_info_keys:
            path["agent_infos"][k] = cur_dist_info[k]


    def compute_and_apply_importance_weights(self,path,age):
        new_logli = self.algo.policy.distribution.log_likelihood(path["actions"],path["agent_infos"])
        logli_diff = new_logli - path["log_likelihood"]
        if self.algo.decision_weight_mode=='pd':
            logli_diff = logli_diff[::-1]
            log_decision_weighted_IS_coeffs = special.discount_cumsum(logli_diff,1)
            IS_coeff = np.exp(log_decision_weighted_IS_coeffs[::-1])
        elif self.algo.decision_weight_mode=='pt':
            IS_coeff = np.exp(np.sum(logli_diff))
        if self.algo.clip_IS_coeff_above:
            IS_coeff = np.minimum(IS_coeff,self.algo.IS_coeff_upper_bound)
        if self.algo.clip_IS_coeff_below:
            IS_coeff = np.maximum(IS_coeff,self.algo.IS_coeff_lower_bound)

        path["weights"] *= IS_coeff

        self.IS_coeffs[age].append(IS_coeff)


    def create_samples_dict(self, paths):
        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
            weights = tensor_utils.concat_tensor_list([path["weights"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                weights=weights,
                paths=paths,
            )

        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            weights = [path["weights"] for path in paths]
            weights = tensor_utils.pad_tensor_n(weights, max_path_length)

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                weights=weights,
                paths=paths,
            )

        return samples_data


    def record_statistics(self, itr, paths, baselines, returns):

        evs = [special.explained_variance_1d(
                    np.concatenate(baselines[i]),
                    np.concatenate(returns[i])
                    ) for i in range(len(baselines))]
        evs = evs[::-1]

        average_discounted_return, undiscounted_returns, ent = self.statistics_for_new_paths()

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', evs[0])
        logger.record_tabular('NumBatches',len(self.experience_replay))
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('MeanPathLen',self.mean_path_len)
        logger.record_tabular('EnvInteracts',self.env_interacts)
        logger.record_tabular('TotalEnvInteracts',self.total_env_interacts)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        if self.algo.batch_aggregate_n > 1:
            """
            for age, raw_weight, weight in zip(self.age, 
                                               self.raw_weights,
                                               self.weights):
                logger.record_tabular('RawWeight_age_' + str(age),raw_weight)
                logger.record_tabular('ScaledWeight_age_' + str(age), weight)
                if age > 0 and self.algo.importance_sampling:
                    IS = tensor_utils.concat_tensor_list(self.IS_coeffs[age])
                    logger.record_tabular('MeanISCoeff_age_' + str(age),np.mean(IS))
                    logger.record_tabular('StdISCoeff_age_' + str(age),np.std(IS))
                    logger.record_tabular('MaxISCoeff_age_' + str(age),np.max(IS))
                    logger.record_tabular('MinISCoeff_age_' + str(age),np.min(IS))
                if age > 0:
                    logger.record_tabular('ExplainedVariance_age_'+str(age),evs[age])
            """
            for age in range(self.algo.batch_aggregate_n):
                if age < len(self.experience_replay):
                    raw_weight = self.raw_weights[::-1][age]
                    weight     = self.weights[::-1][age]
                    logger.record_tabular('RawWeight_age_' + str(age),raw_weight)
                    logger.record_tabular('ScaledWeight_age_' + str(age),weight)
                    if age > 0 and self.algo.importance_sampling:
                        IS = self.get_IS(age)
                        logger.record_tabular('MeanISCoeff_age_' + str(age),np.mean(IS))
                        logger.record_tabular('StdISCoeff_age_' + str(age),np.std(IS))
                        logger.record_tabular('MaxISCoeff_age_' + str(age),np.max(IS))
                        logger.record_tabular('MinISCoeff_age_' + str(age),np.min(IS))
                    logger.record_tabular('ExplainedVariance_age_'+str(age),evs[age])
                else:
                    logger.record_tabular('RawWeight_age_' + str(age),0)
                    logger.record_tabular('ScaledWeight_age_' + str(age),0)
                    if age > 0 and self.algo.importance_sampling:
                        logger.record_tabular('MeanISCoeff_age_' + str(age),0)
                        logger.record_tabular('StdISCoeff_age_' + str(age),0)
                        logger.record_tabular('MaxISCoeff_age_' + str(age),0)
                        logger.record_tabular('MinISCoeff_age_' + str(age),0)
                    logger.record_tabular('ExplainedVariance_age_'+str(age),0)
                

        if self.algo.exploration_bonus:
            bonuses = tensor_utils.concat_tensor_list([path["bonuses"] for path in paths])
            logger.record_tabular('MeanRawBonus',self.bonus_mean)
            logger.record_tabular('MeanBonus',np.mean(bonuses))
            logger.record_tabular('StdBonus',np.std(bonuses))
            logger.record_tabular('MaxBonus',np.max(bonuses))
            bonus_sums = np.array([np.sum(path["bonuses"]) for path in paths])
            logger.record_tabular('MeanBonusSum', np.mean(bonus_sums))
            logger.record_tabular('StdBonusSum', np.std(bonus_sums))
            if self.algo.batch_aggregate_n > 1:
                new_bonuses = tensor_utils.concat_tensor_list(
                            [path["bonuses"] for path in self.experience_replay[-1]]
                            )
                logger.record_tabular('NewPathsMeanBonus',np.mean(new_bonuses))
                logger.record_tabular('NewPathsStdBonus',np.std(new_bonuses))
                logger.record_tabular('NewPathsMaxBonus',np.max(new_bonuses))

    def get_IS(self,age):
        if self.algo.decision_weight_mode=='pd':
            return tensor_utils.concat_tensor_list(self.IS_coeffs[age])
        else:
            return np.array(self.IS_coeffs[age])


    def statistics_for_new_paths(self):
        average_discounted_return = \
            np.mean([path["returns"][0] for path in self.experience_replay[-1]])

        undiscounted_returns = [sum(path["rewards"]) for path in self.experience_replay[-1]]

        agent_infos = tensor_utils.concat_tensor_dict_list( 
                            [path["agent_infos"] for path in self.experience_replay[-1]]
                            )
        ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

        return average_discounted_return, undiscounted_returns, ent


    def update_parametrized_models(self):
        """only most recent batch of data is used to fit models"""

        logger.log("fitting baseline...")
        self.algo.baseline.fit(self.experience_replay[-1])
        logger.log("fitted")

        if self.algo.exploration_bonus:
            logger.log("fitting exploration bonus model...")
            self.algo.exploration_bonus.fit(self.experience_replay[-1])
            logger.log("fitted")
