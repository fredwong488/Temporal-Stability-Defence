from .pointpillars import PointPillarsDetector
from .pointpillars_nuscenes import PointPillarsNuScenesDetector
from .pointrcnn import PointRCNNDetector
from .precomputed import PrecomputedDetector

__all__ = ["PointPillarsDetector", "PointPillarsNuScenesDetector", "PointRCNNDetector", "PrecomputedDetector"]
