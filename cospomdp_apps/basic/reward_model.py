# Reward model for 2D

from cospomdp.models.reward_model import ObjectSearchRewardModel

class ObjectSearchRewardModel2D(ObjectSearchRewardModel):
    def __init__(self, goal_dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_dist = goal_dist
