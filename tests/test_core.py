import pytest
from unittest.mock import Mock, patch
from jetski_tracker.core import (
    JetSkiTrackingApp,
    YOLOModelTrainer,
    DatasetValidator,
    ModelRepository,
    ConfigManager
)
from jetski_tracker.exceptions import (
    InvalidDatasetError,
    ModelDownloadError,
    ConfigurationError
)

class TestJetSkiTrackingApp:
    def test_execute_workflow_valid(self, mock_config):
        mock_trainer = Mock()
        mock_processor = Mock()
        mock_validator = Mock(return_value=True)
        mock_config_manager = Mock()
        mock_config_manager.config = mock_config
        
        app = JetSkiTrackingApp(
            mock_trainer,
            mock_processor,
            mock_validator,
            mock_config_manager
        )
        
        app.execute_workflow("test_video.mp4")
        mock_trainer.train.assert_called_once_with(mock_config)
        
    def test_execute_workflow_invalid_dataset(self, mock_config):
        mock_validator = Mock(return_value=False)
        
        with pytest.raises(InvalidDatasetError):
            app = JetSkiTrackingApp(
                Mock(),
                Mock(),
                mock_validator,
                Mock()
            )
            app.execute_workflow("test_video.mp4")

class TestModelRepository:
    @patch('ultralytics.YOLO')
    def test_load_local_model(self, mock_yolo):
        repo = ModelRepository('/tmp')
        repo.load_model('local.pt')
        mock_yolo.assert_called_once_with('local.pt')

    @patch('urllib.request.urlretrieve')
    def test_download_model_failure(self, mock_retrieve):
        mock_retrieve.side_effect = Exception("Download failed")
        repo = ModelRepository('/tmp')
        
        with pytest.raises(ModelDownloadError):
            repo._download_model('http://example.com/model.pt', '/tmp/model.pt')

class TestConfigManager:
    def test_config_property_returns_copy(self, mock_config):
        manager = ConfigManager(mock_config)
        config = manager.config
        config['new_key'] = 'value'
        assert 'new_key' not in manager.config