"""
JetSki Tracking System - Computer vision system for detecting and tracking jetskis in video streams.
"""

from .core import (
    JetSkiTrackingApp,
    YOLOModelTrainer,
    JetSkiVideoProcessor,
    DatasetValidator,
    ModelRepository,
    ConfigManager
)
from .exceptions import (
    JetSkiTrackingError,
    ConfigurationError,
    VideoProcessingError,
    ModelDownloadError,
    InvalidDatasetError
)

__version__ = "0.1.0"
__all__ = [
    'JetSkiTrackingApp',
    'bootstrap_application',
    'JetSkiTrackingError',
    'ConfigurationError',
    'VideoProcessingError'
]