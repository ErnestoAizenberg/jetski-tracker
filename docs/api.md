# API Reference

## Core Classes

### `JetSkiTrackingApp`

Main application class that orchestrates the tracking workflow.

#### Methods:
- `execute_workflow(video_source: str) -> None`: Runs full processing pipeline
- `with_config(config: dict) -> JetSkiTrackingApp`: Factory method with custom config

### `YOLOModelTrainer`

Handles model training operations.

#### Methods:
- `train(config: dict) -> YOLO`: Trains model with given configuration

## Configuration

### `ConfigManager`

Manages application configuration.

#### Properties:
- `config: dict`: Current configuration (read-only copy)

#### Methods:
- `update(updates: dict) -> None`: Merge in new configuration values
- `save(path: str) -> None`: Save config to YAML file