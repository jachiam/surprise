import numpy as np
import lasagne.nonlinearities as NL
from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.misc import logger

from sandbox.surprisebased.regressors.regularized_gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.surprisebased.exploration_bonuses.base import ParameterizedExplorationBonus

# learns a dynamics model by solving
#
#   max_{phi} sum_i log p_{phi} (s'_i | s_i, a_i),
#
# gives bonuses of the form
#
#   - log p_{phi} (s' | s,a).

class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        #assert self._size > batch_size
        if self._size <= batch_size:
            return dict(
                observations=self._observations[:-1],
                actions=self._actions[:-1],
                next_observations=self._observations[1:]
            )
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample; also, if it's a terminal state, not valid either
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            if self._terminals[index]:
                 continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size



class SurprisalBonus(ParameterizedExplorationBonus, Serializable):

    def __init__(
            self,
            env_spec,
            use_replay_pool=True,
            batch_size=10000,
            max_pool_size=100000,
            weight_decay=0,
            hidden_sizes=(32,32),
            hidden_nonlinearity=NL.rectify,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(SurprisalBonus, self).__init__()
        self.use_replay_pool=use_replay_pool
        self.replay_pool = SimpleReplayPool(
                max_pool_size,
                env_spec.observation_space.flat_dim,
                env_spec.action_space.flat_dim
        )
        self.batch_size = batch_size
        self._weight_decay = weight_decay
        if regressor_args is None:
            regressor_args = dict()
        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim,),
            output_dim=env_spec.observation_space.flat_dim,
            name="dynamics_model",
            adaptive_std=True,
            weight_decay=weight_decay,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            std_hidden_sizes=hidden_sizes,
            std_nonlinearity=hidden_nonlinearity,
            normalize_inputs=False,
            normalize_outputs=False,
            **regressor_args
        )

    def fit(self, paths):
        # stack the observations and actions for each input
        if not(self.use_replay_pool):
            xs = np.concatenate(
                    [np.concatenate(
                            [ p["observations"][:-1] , p["actions"][:-1] ],
                            axis=1) 
                    for p in paths]
                    )
            ys = np.concatenate([p["observations"][1:] for p in paths])
            self._regressor.fit(xs,ys)
        else:
            for p in paths:
                for t in range(len(p["observations"])):
                    if t==len(p["observations"])-1:
                        terminal=True
                    else:
                        terminal=False
                    self.replay_pool.add_sample(p["observations"][t], p["actions"][t],terminal)
            batch = self.replay_pool.random_batch(self.batch_size)
            xs = np.concatenate([batch["observations"],batch["actions"]],axis=1)
            ys = batch["next_observations"]
            self._regressor.fit(xs,ys)

    def predict(self, path):
        return self._regressor.predict(
                        np.concatenate([path["observations"], path["actions"]], axis=1) 
                        )

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def get_log_probs(self,path):
        xs = np.concatenate( [ path["observations"][:-1], path["actions"][:-1] ], axis=1)
        ys = path["observations"][1:]
        return self._regressor.predict_log_likelihood(xs, ys)
       

    def get_bonus(self, path):
        logli = -self.get_log_probs(path)
        logli = np.append(logli,[0])
        return logli

