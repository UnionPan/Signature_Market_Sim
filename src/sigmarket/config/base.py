"""Base configuration classes."""

from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path
import yaml


@dataclass
class BaseConfig:
    """Base configuration class with serialization support."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> None:
        """Validate configuration parameters.

        Override this method in subclasses to add validation logic.
        Raises ValueError if validation fails.
        """
        pass
