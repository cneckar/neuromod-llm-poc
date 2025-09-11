"""
Neuromodulation Testing Framework
"""

from .base_test import BaseTest
from .simple_emotion_tracker import SimpleEmotionTracker
from .test_runner import TestRunner
from .test_suite import TestSuite
from .story_emotion_test import StoryEmotionTest
from .adq_test import ADQTest
from .cdq_test import CDQTest
from .sdq_test import SDQTest
from .ddq_test import DDQTest
from .pdq_test import PDQTest
from .edq_test import EDQTest
from .pcq_pop_test import PCQPopTest
from .cognitive_tasks import CognitiveTasksTest

# Import new scientific framework components
from .telemetry import TelemetryCollector, TelemetryMetrics, TelemetrySummary
from .experimental_design import ExperimentalDesigner, ExperimentalSession, ConditionType

# Import visualization and results components
from .visualization import NeuromodVisualizer
from .results_templates import (
    ResultsTemplateGenerator, 
    StatisticalResult, 
    PackResult, 
    ModelResult
)

# Import advanced statistics components
from .advanced_statistics import (
    AdvancedStatisticalAnalyzer,
    MixedEffectsResult,
    BayesianResult,
    CanonicalCorrelationResult
)

# Import human reference data components
from .human_reference_worksheets import HumanReferenceDataCollector
from .signature_matching import (
    SignatureMatcher, 
    HumanSignature, 
    ModelSignature, 
    SignatureMatch
)
from .human_reference_workbook import HumanReferenceWorkbook

__all__ = [
    'BaseTest',
    'SimpleEmotionTracker',
    'TestRunner',
    'TestSuite', 
    'StoryEmotionTest',
    'ADQTest',
    'CDQTest',
    'SDQTest',
    'DDQTest',
    'PDQTest',
    'EDQTest',
    'PCQPopTest',
    'CognitiveTasksTest',
    
    # Scientific framework
    'TelemetryCollector', 'TelemetryMetrics', 'TelemetrySummary',
    'ExperimentalDesigner', 'ExperimentalSession', 'ConditionType',
    
    # Visualization and results
    'NeuromodVisualizer',
    'ResultsTemplateGenerator', 'StatisticalResult', 'PackResult', 'ModelResult',
    
    # Advanced statistics
    'AdvancedStatisticalAnalyzer', 'MixedEffectsResult', 'BayesianResult', 'CanonicalCorrelationResult',
    
    # Human reference data
    'HumanReferenceDataCollector', 'SignatureMatcher', 'HumanSignature', 'ModelSignature', 'SignatureMatch', 'HumanReferenceWorkbook'
]
