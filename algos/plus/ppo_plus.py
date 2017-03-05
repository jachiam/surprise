from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.jachiam.algos.plus.policy_gradient_plus import PolicyGradientPlus
from rllab.core.serializable import Serializable


class PPO(PolicyGradientPlus, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
