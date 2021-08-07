# The POMDP-related components for Thor Object Search
from ..models import cospomdp

# COS-POMDP inheritences
class HighLevelSearchRegion(cospomdp.SearchRegion):
    def __init__(self):
        pass

class LowLevelSearchRegion(cospomdp.SearchRegion):
    def __init__(self):
        pass

class COSPTransitionModel(cospomdp.TransitionModel):
    def __init__(self):
        pass

class COSPDetectionModel(cospomdp.DetectionModel):
    """Interface for Pr(zi | si, srobot'); Domain-specific"""
    def __init__(self):
        pass
