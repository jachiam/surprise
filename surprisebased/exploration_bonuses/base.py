from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable


class ExplorationBonus(object):

    def get_bonus(self, paths):
        raise NotImplementedError

    def fit(self, paths):
        raise NotImplementedError


class ParameterizedExplorationBonus(Parameterized):

    def __init__(self, **kwargs):
        super(ParameterizedExplorationBonus,self).__init__(**kwargs)

    def get_bonus(self, paths):
        raise NotImplementedError

    def fit(self, paths):
        raise NotImplementedError
