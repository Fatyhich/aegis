"""
Configuration loader for YAML files with Pydantic validation.
"""

from pathlib import Path
from typing import Tuple
import yaml

from .schema import PipelineConfig, ModelsConfig, GraphConfig


def load_yaml(file_path: Path) -> dict:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """
    Load and validate pipeline configuration.

    Args:
        config_path: Path to pipeline.yaml

    Returns:
        PipelineConfig instance
    """
    data = load_yaml(Path(config_path))
    return PipelineConfig(**data)


def load_models_config(config_path: str | Path) -> ModelsConfig:
    """
    Load and validate models configuration.

    Args:
        config_path: Path to models.yaml

    Returns:
        ModelsConfig instance
    """
    data = load_yaml(Path(config_path))
    return ModelsConfig(**data)


def load_graph_config(config_path: str | Path) -> GraphConfig:
    """
    Load and validate graph configuration.

    Args:
        config_path: Path to graph.yaml

    Returns:
        GraphConfig instance
    """
    data = load_yaml(Path(config_path))
    return GraphConfig(**data)


def load_all_configs(config_dir: str | Path) -> Tuple[PipelineConfig, ModelsConfig, GraphConfig]:
    """
    Load all configuration files from a directory.

    Args:
        config_dir: Directory containing YAML config files

    Returns:
        Tuple of (pipeline_config, models_config, graph_config)
    """
    config_dir = Path(config_dir)

    pipeline_config = load_pipeline_config(config_dir / "pipeline.yaml")
    models_config = load_models_config(config_dir / "models.yaml")
    graph_config = load_graph_config(config_dir / "graph.yaml")

    return pipeline_config, models_config, graph_config
