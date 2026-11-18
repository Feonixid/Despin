"""
DESPIN: Dynamic Epistemic Slicing and Pivot-Initiated Navigation
A training-free decoding framework for trajectory rescue in reasoning LLMs.
"""

from .detector import AttractorDetector, LoopDetectionResult
from .detector_v2 import PhaseSpaceRecurrenceDetector, PhaseSpaceDetectionResult
from .kv_cache import KVCacheManager
from .engine import DespinEngine
from .engine_v2 import DespinEngineV2, GenerationResultV2
from .evaluator import MathEvaluator

__version__ = "0.2.0"
__all__ = [
    "AttractorDetector",
    "LoopDetectionResult",
    "PhaseSpaceRecurrenceDetector",
    "PhaseSpaceDetectionResult",
    "KVCacheManager",
    "DespinEngine",
    "DespinEngineV2",
    "GenerationResultV2",
    "MathEvaluator",
]
