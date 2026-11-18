"""DESPIN: phase-space recurrence detection, KV-cache rollback and attractor-subspace deflation.

Modules are imported explicitly so that the evaluator can be used without torch:

    from despin.engine import DespinEngineV2          # needs torch + transformers
    from despin.evaluator import RobustMathEvaluator  # pure Python
"""
__version__ = "0.2.0"
