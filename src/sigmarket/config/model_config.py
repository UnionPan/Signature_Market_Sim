"""Model configuration classes."""

from dataclasses import dataclass
from typing import Optional
from .base import BaseConfig


@dataclass
class CVAEConfig(BaseConfig):
    """Configuration for Conditional Variational Autoencoder.

    This matches the architecture from the original TensorFlow implementation.

    Attributes:
        n_latent: Dimension of latent space
        n_hidden: Number of units in hidden layers
        alpha: Weight for KL divergence term (0 < alpha < 1)
              Loss = (1-alpha) * reconstruction + alpha * KL
        activation: Activation function ('leaky_relu', 'relu', 'elu', 'tanh')
        leaky_relu_alpha: Alpha parameter for LeakyReLU (if used)
        dropout: Dropout rate (0 for no dropout)
        use_batch_norm: Whether to use batch normalization
    """

    n_latent: int = 8
    n_hidden: int = 50
    alpha: float = 0.003
    activation: str = 'leaky_relu'
    leaky_relu_alpha: float = 0.3
    dropout: float = 0.0
    use_batch_norm: bool = False

    def validate(self) -> None:
        """Validate CVAE configuration."""
        if self.n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {self.n_latent}")

        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {self.n_hidden}")

        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        valid_activations = ['leaky_relu', 'relu', 'elu', 'tanh', 'sigmoid']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")

        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for Transformer-based generative model.

    Attributes:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Dimension of feedforward network
        dropout: Dropout rate
        activation: Activation function for FFN
        use_positional_encoding: Whether to use positional encoding
        max_seq_len: Maximum sequence length for positional encoding
    """

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = 'gelu'
    use_positional_encoding: bool = True
    max_seq_len: int = 1000

    def validate(self) -> None:
        """Validate Transformer configuration."""
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {self.d_model} and {self.n_heads}")

        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")

        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class DiffusionConfig(BaseConfig):
    """Configuration for Diffusion-based generative model.

    Attributes:
        timesteps: Number of diffusion timesteps
        beta_schedule: Schedule for noise addition ('linear', 'cosine', 'quadratic')
        beta_start: Starting value for beta schedule
        beta_end: Ending value for beta schedule
        model_channels: Base number of channels in U-Net
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: Resolutions to apply attention at
        channel_mult: Channel multiplier for each level
        dropout: Dropout rate
        use_ema: Whether to use exponential moving average of weights
        ema_decay: Decay rate for EMA
    """

    timesteps: int = 1000
    beta_schedule: str = 'cosine'
    beta_start: float = 0.0001
    beta_end: float = 0.02
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)
    channel_mult: tuple = (1, 2, 4, 8)
    dropout: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.9999

    def validate(self) -> None:
        """Validate Diffusion configuration."""
        if self.timesteps < 1:
            raise ValueError(f"timesteps must be >= 1, got {self.timesteps}")

        valid_schedules = ['linear', 'cosine', 'quadratic']
        if self.beta_schedule not in valid_schedules:
            raise ValueError(f"beta_schedule must be one of {valid_schedules}, got {self.beta_schedule}")

        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError(f"Must have 0 < beta_start < beta_end < 1, got {self.beta_start} and {self.beta_end}")

        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        if not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")


@dataclass
class ModelConfig(BaseConfig):
    """Unified model configuration wrapper.

    This allows for easy model selection and configuration.

    Attributes:
        model_type: Type of model ('cvae', 'transformer', 'diffusion')
        cvae_config: Configuration for CVAE (if model_type='cvae')
        transformer_config: Configuration for Transformer (if model_type='transformer')
        diffusion_config: Configuration for Diffusion (if model_type='diffusion')
    """

    model_type: str = 'cvae'
    cvae_config: Optional[CVAEConfig] = None
    transformer_config: Optional[TransformerConfig] = None
    diffusion_config: Optional[DiffusionConfig] = None

    def __post_init__(self):
        """Initialize specific config based on model_type."""
        if self.model_type == 'cvae' and self.cvae_config is None:
            self.cvae_config = CVAEConfig()
        elif self.model_type == 'transformer' and self.transformer_config is None:
            self.transformer_config = TransformerConfig()
        elif self.model_type == 'diffusion' and self.diffusion_config is None:
            self.diffusion_config = DiffusionConfig()

    def get_config(self):
        """Get the active model configuration."""
        if self.model_type == 'cvae':
            return self.cvae_config
        elif self.model_type == 'transformer':
            return self.transformer_config
        elif self.model_type == 'diffusion':
            return self.diffusion_config
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def validate(self) -> None:
        """Validate model configuration."""
        valid_types = ['cvae', 'transformer', 'diffusion']
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {self.model_type}")

        # Validate specific config
        config = self.get_config()
        if config is not None:
            config.validate()
