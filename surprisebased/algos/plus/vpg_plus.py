from sandbox.jachiam.algos.plus.policy_gradient_plus import PolicyGradientPlus
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.serializable import Serializable


class VPG(PolicyGradientPlus, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.opt_info = None
        super(VPG, self).__init__(env=env, policy=policy, baseline=baseline, optimizer=optimizer, **kwargs)
