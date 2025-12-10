"""Configuration loader with validation and environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML file with environment variable overrides."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config loader.
        
        Args:
            config_path: Path to config.yaml file. If None, looks for config/config.yaml
        """
        # Load environment variables
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        load_dotenv(dotenv_path=env_path)
        
        # Find config file
        if config_path is None:
            config_path = project_root / "config" / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML config
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Model overrides
        if os.getenv("HF_MODEL_NAME"):
            self._config["model"]["name"] = os.getenv("HF_MODEL_NAME")
        
        if os.getenv("HF_DEVICE"):
            self._config["model"]["device"] = os.getenv("HF_DEVICE")
        
        # UI overrides
        if os.getenv("GRADIO_PORT"):
            self._config["ui"]["port"] = int(os.getenv("GRADIO_PORT"))
        
        if os.getenv("GRADIO_SHARE"):
            self._config["ui"]["share"] = os.getenv("GRADIO_SHARE").lower() == "true"
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., "model.name")
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key_path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get("model", {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self._config.get("retrieval", {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self._config.get("vector_store", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get("ui", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data paths configuration."""
        return self._config.get("data", {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config


def load_config(config_path: Optional[Path] = None) -> ConfigLoader:
    """Load configuration from file.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)

