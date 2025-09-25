"""
AD Data Analysis Framework
=========================

A comprehensive framework for data analysis and interpretation.

Main modules:
- core: Core data processing and analysis functions
- visualization: Data visualization utilities
- interpretation: Data interpretation and insights generation
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "AD Team"

from .core import DataProcessor, StatisticalAnalyzer
from .visualization import DataVisualizer
from .interpretation import DataInterpreter
from .utils import DataLoader, DataValidator

__all__ = [
    'DataProcessor',
    'StatisticalAnalyzer',
    'DataVisualizer',
    'DataInterpreter',
    'DataLoader',
    'DataValidator'
]