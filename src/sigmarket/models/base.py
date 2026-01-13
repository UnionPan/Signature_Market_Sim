"""Base class for all generative models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class BaseGenerativeModel(nn.Module, ABC):
    """Abstract base class for all generative models.

    All generative models (CVAE, Transformer, Diffusion) must implement this interface.
    This ensures consistency across different model architectures and makes them
    interchangeable in the training and generation pipelines.

    Methods to implement:
        - forward: Forward pass through the model
        - generate: Generate new samples
        - compute_loss: Compute model-specific loss

    Optional methods to override:
        - encode: Encode input to latent representation
        - decode: Decode from latent representation
        - configure_optimizers: Set up optimizers and schedulers
    """

    def __init__(self):
        super().__init__()
        self._input_dim = None
        self._condition_dim = None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            condition: Optional condition tensor of shape [batch_size, condition_dim]
            **kwargs: Additional model-specific arguments

        Returns:
            Output tensor (model-specific shape and meaning)
        """
        pass

    @abstractmethod
    def generate(
        self,
        condition: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Generate new samples.

        Args:
            condition: Optional condition tensor of shape [n_samples, condition_dim]
                      or [condition_dim] (will be expanded to n_samples)
            n_samples: Number of samples to generate
            **kwargs: Additional generation parameters (e.g., temperature, steps)

        Returns:
            Generated samples of shape [n_samples, output_dim]
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute model-specific loss for training.

        Args:
            batch: Dictionary containing:
                - 'x': Input data [batch_size, input_dim]
                - 'condition': Condition data [batch_size, condition_dim]
                - Other model-specific keys
            **kwargs: Additional loss computation arguments

        Returns:
            loss: Total loss tensor (scalar)
            metrics: Dictionary of component losses for logging
                    e.g., {'total_loss': 0.5, 'recon_loss': 0.3, 'kl_loss': 0.2}
        """
        pass

    def encode(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input to latent representation.

        Optional method - only implement if model has explicit encoding step.

        Args:
            x: Input tensor
            condition: Optional condition tensor

        Returns:
            Latent representation
        """
        raise NotImplementedError("Encode method not implemented for this model")

    def decode(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode from latent representation.

        Optional method - only implement if model has explicit decoding step.

        Args:
            z: Latent tensor
            condition: Optional condition tensor

        Returns:
            Decoded output
        """
        raise NotImplementedError("Decode method not implemented for this model")

    def save(self, path: str) -> None:
        """Save model weights and configuration.

        Args:
            path: Path to save model (should end in .pt or .pth)
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'input_dim': self._input_dim,
            'condition_dim': self._condition_dim,
        }

        torch.save(save_dict, save_path)

    def load(self, path: str, device: str = 'cpu') -> None:
        """Load model weights.

        Args:
            path: Path to saved model
            device: Device to load model to
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._input_dim = checkpoint.get('input_dim')
        self._condition_dim = checkpoint.get('condition_dim')

    def configure_optimizers(
        self,
        optimizer_name: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Configure optimizer for training.

        Can be overridden for model-specific optimizer configuration.

        Args:
            optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            **kwargs: Additional optimizer arguments

        Returns:
            Configured optimizer
        """
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_input_dim(self, input_dim: int) -> None:
        """Set input dimension (called during initialization)."""
        self._input_dim = input_dim

    def set_condition_dim(self, condition_dim: Optional[int]) -> None:
        """Set condition dimension (called during initialization)."""
        self._condition_dim = condition_dim

    @property
    def input_dim(self) -> Optional[int]:
        """Get input dimension."""
        return self._input_dim

    @property
    def condition_dim(self) -> Optional[int]:
        """Get condition dimension."""
        return self._condition_dim

    def __repr__(self) -> str:
        """String representation of model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  input_dim={self._input_dim},\n"
            f"  condition_dim={self._condition_dim},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )
