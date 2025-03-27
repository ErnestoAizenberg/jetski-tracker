import pytest
from unittest.mock import Mock
from jetski_tracker.core import ConfigManager
import tempfile

@pytest.fixture
def mock_config():
    return {
        'base_model': 'yolov8n.pt',
        'data_config': 'config.yaml',
        'data_path': 'datasets',
        'epochs': 1,
        'imgsz': 320,
        'tracker': {
            'persist': True,
            'classes': [0]
        }
    }

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def config_manager(mock_config):
    return ConfigManager(mock_config)