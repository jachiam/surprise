from rllab.core.serializable import Serializable
from sandbox.jachiam.exploration_bonuses.base import ExplorationBonus

class NoBonus(ExplorationBonus,Serializable):

    def __init__(self):
        Serializable.quick_init(self, locals())

    def get_bonus(self, paths):
        return 0

    def fit(self,paths):
        return
