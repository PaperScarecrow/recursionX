"""Recursion-X: a looped liquid-hybrid MoE transformer with test-time memory,
hashed n-gram tables, projected-LoRA skill acquisition and dual-hemisphere
sleep consolidation."""
from .config import RXConfig
from .model import RecursionX, RXOutput
from .lifecycle.config import LifecycleConfig
from .lifecycle.brain import DualHemisphereBrain
from .lifecycle.skills import SkillRecord, SkillRouter

__all__ = ["RXConfig", "RecursionX", "RXOutput", "LifecycleConfig", "DualHemisphereBrain",
           "SkillRecord", "SkillRouter"]
__version__ = "0.1.0"
