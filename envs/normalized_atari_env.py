import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step


class NormalizedAtariEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._wrapped_env = env

    def _apply_normalize_obs(self, obs):
        return (obs/128. - 1)/3

    def reset(self):
        ret = self._wrapped_env.reset()
        return self._apply_normalize_obs(ret)

    @overrides
    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        next_obs = self._apply_normalize_obs(next_obs)
        return Step(next_obs, reward, done, **info)

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

normalize = NormalizedAtariEnv
