"""
AIAS - AI Assistant System
Core package initialization
"""

__version__ = "1.0.0"
__author__ = "AIAS Development"

from .orchestrator import AIASOrchestrator
from .groq_llm import GroqVisionLLM

__all__ = ["AIASOrchestrator", "GroqVisionLLM"]
