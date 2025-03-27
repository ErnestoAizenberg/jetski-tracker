class JetSkiTrackingError(Exception):
    """Base exception for all package-specific errors"""
    pass

class ConfigurationError(JetSkiTrackingError):
    """Invalid configuration provided"""
    pass

class VideoProcessingError(JetSkiTrackingError):
    """Video frame processing failure"""
    pass

class ModelDownloadError(JetSkiTrackingError):
    """Failed to download model"""
    pass

class InvalidDatasetError(JetSkiTrackingError):
    """Dataset validation failed"""
    pass

class FrameProcessingError(VideoProcessingError):
    """Error during individual frame processing"""
    pass