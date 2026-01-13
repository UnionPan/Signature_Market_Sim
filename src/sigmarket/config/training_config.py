"""Training pipeline configuration."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training pipeline.

    Attributes:
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization)
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: LR scheduler type ('cosine', 'step', 'plateau', None)
        scheduler_params: Additional parameters for scheduler
        gradient_clip: Maximum gradient norm (None for no clipping)
        accumulate_grad_batches: Number of batches to accumulate gradients over
        device: Device to train on ('cuda', 'cpu', 'auto')
        use_amp: Whether to use automatic mixed precision
        seed: Random seed for reproducibility
        checkpoint_dir: Directory to save checkpoints
        save_top_k: Number of best checkpoints to keep
        early_stopping: Whether to use early stopping
        early_stopping_patience: Number of epochs to wait for improvement
        early_stopping_min_delta: Minimum change to qualify as improvement
        log_every_n_steps: Log metrics every N steps
        val_check_interval: Validate every N training steps
        callbacks: List of callback names to use
    """

    # Training hyperparameters
    n_epochs: int = 10000
    batch_size: int = 32
    learning_rate: float = 0.005
    weight_decay: float = 0.0

    # Optimizer
    optimizer: str = 'adam'

    # Learning rate scheduler
    scheduler: Optional[str] = None
    scheduler_params: dict = field(default_factory=dict)

    # Gradient handling
    gradient_clip: Optional[float] = None
    accumulate_grad_batches: int = 1

    # Device and performance
    device: str = 'auto'
    use_amp: bool = False
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_top_k: int = 3
    save_last: bool = True

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    # Logging
    log_every_n_steps: int = 10
    val_check_interval: int = 100

    # Callbacks
    callbacks: List[str] = field(default_factory=lambda: ['checkpoint'])

    def validate(self) -> None:
        """Validate training configuration."""
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

        valid_optimizers = ['adam', 'adamw', 'sgd']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got {self.optimizer}")

        valid_schedulers = ['cosine', 'step', 'plateau', None]
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got {self.scheduler}")

        valid_devices = ['cuda', 'cpu', 'auto']
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")

        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be > 0, got {self.gradient_clip}")

        if self.accumulate_grad_batches < 1:
            raise ValueError(f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}")

        if self.save_top_k < 0:
            raise ValueError(f"save_top_k must be >= 0, got {self.save_top_k}")

        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}")

        if self.log_every_n_steps < 1:
            raise ValueError(f"log_every_n_steps must be >= 1, got {self.log_every_n_steps}")

        if self.val_check_interval < 1:
            raise ValueError(f"val_check_interval must be >= 1, got {self.val_check_interval}")

    def get_checkpoint_path(self) -> Path:
        """Get checkpoint directory path."""
        return Path(self.checkpoint_dir).expanduser()

    def create_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        checkpoint_path = self.get_checkpoint_path()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
