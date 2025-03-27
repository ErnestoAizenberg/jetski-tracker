from abc import ABC, abstractmethod
import cv2
import os
import yaml
import urllib.request
import tempfile
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from ultralytics import YOLO
from .exceptions import (
    ModelDownloadError,
    InvalidDatasetError,
    FrameProcessingError,
    ConfigurationError
)

class IModelTrainer(ABC):
    @abstractmethod
    def train(self, config: Dict[str, Any]) -> Any:
        """Train a model with given configuration"""
        pass

class IVideoProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame: Any) -> Any:
        """Process a single video frame"""
        pass

class IDataValidator(ABC):
    @abstractmethod
    def validate(self, data_path: str) -> bool:
        """Validate dataset structure"""
        pass

class YOLOModelTrainer(IModelTrainer):
    def __init__(self, model_repository: 'ModelRepository'):
        self._repo = model_repository

    def train(self, config: Dict[str, Any]) -> YOLO:
        try:
            model = self._repo.load_model(config['base_model'])
            return model.train(
                data=config['data_config'],
                epochs=config['epochs'],
                imgsz=config['imgsz'],
                batch=config.get('batch', 8),
                name=config.get('name', 'jetski_detection')
            )
        except Exception as e:
            raise ConfigurationError(f"Training failed: {str(e)}")

class JetSkiVideoProcessor(IVideoProcessor):
    def __init__(self, model: YOLO, tracker_config: Dict[str, Any]):
        self._model = model
        self._config = tracker_config

    def process_frame(self, frame: Any) -> Any:
        try:
            results = self._model.track(
                frame,
                persist=self._config['persist'],
                classes=self._config['classes'],
                verbose=False
            )
            return results[0].plot()
        except Exception as e:
            raise FrameProcessingError(f"Frame processing failed: {str(e)}")

class DatasetValidator(IDataValidator):
    def validate(self, data_path: str) -> bool:
        required_dirs = {'images', 'labels'}
        try:
            present_dirs = set(os.listdir(data_path))
            return required_dirs.issubset(present_dirs)
        except FileNotFoundError:
            return False

class ModelRepository:
    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_spec: str) -> YOLO:
        if model_spec.startswith('http'):
            return self._load_remote_model(model_spec)
        return YOLO(model_spec)

    def _load_remote_model(self, url: str) -> YOLO:
        model_path = self._cache_dir / Path(url).name
        if not model_path.exists():
            self._download_model(url, model_path)
        return YOLO(str(model_path))

    @staticmethod
    def _download_model(url: str, dest: Path):
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            raise ModelDownloadError(f"Failed to download model: {str(e)}")

class ConfigManager:
    def __init__(self, base_config: Dict[str, Any]):
        self._config = base_config

    def update(self, updates: Dict[str, Any]) -> None:
        self._config.update(updates)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self._config, f)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

class JetSkiTrackingApp:
    def __init__(
        self,
        model_trainer: IModelTrainer,
        video_processor: IVideoProcessor,
        data_validator: IDataValidator,
        config_manager: ConfigManager
    ):
        self._trainer = model_trainer
        self._processor = video_processor
        self._validator = data_validator
        self._config = config_manager

    def execute_workflow(self, video_source: str) -> None:
        if not self._validator.validate(self._config.config['data_path']):
            raise InvalidDatasetError("Invalid dataset structure")

        trained_model = self._trainer.train(self._config.config)
        self._process_video(video_source, trained_model)

    def _process_video(self, source: str, model: YOLO) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video source: {source}")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed = self._processor.process_frame(frame)
                cv2.imshow("JetSki Tracking", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

def create_default_config(temp_dir: str) -> ConfigManager:
    base_config = {
        'base_model': 'yolov8n.pt',
        'data_config': str(Path(temp_dir) / 'config.yaml'),
        'data_path': str(Path(temp_dir) / 'datasets'),
        'epochs': 10,
        'imgsz': 640,
        'tracker': {
            'persist': True,
            'classes': [0]
        }
    }
    return ConfigManager(base_config)

def bootstrap_application() -> JetSkiTrackingApp:
    temp_dir = tempfile.mkdtemp(prefix='jetski_')
    model_repo = ModelRepository(temp_dir)
    config = create_default_config(temp_dir)
    
    (Path(temp_dir) / 'datasets' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(temp_dir) / 'datasets' / 'labels').mkdir(parents=True, exist_ok=True)
    config.save(config.config['data_config'])

    return JetSkiTrackingApp(
        model_trainer=YOLOModelTrainer(model_repo),
        video_processor=JetSkiVideoProcessor(
            model=model_repo.load_model(config.config['base_model']),
            tracker_config=config.config['tracker']
        ),
        data_validator=DatasetValidator(),
        config_manager=config
    )