"""
SigMarket: PyTorch-based Signature-Method Market Simulation Library

A modern, modular library for generating synthetic financial market data using
signature-based methods and deep generative models.

Based on the paper:
    H. Buehler, B. Horvath, T. Lyons, I. Perez Arribas and B. Wood.
    "Generating financial markets with signatures." SSRN 3657366, 2020.

Main Components:
    - Data Pipeline: Load, preprocess, and compute signatures
    - Training Pipeline: Train generative models (CVAE, Transformer, Diffusion)
    - Generation Pipeline: Generate synthetic paths and validate

Example Usage:
    >>> from sigmarket import DataPipeline, Trainer, MarketGenerator
    >>> from sigmarket.config import DataPipelineConfig, CVAEConfig, TrainingConfig
    >>>
    >>> # Configure and prepare data
    >>> data_config = DataPipelineConfig(ticker='AAPL', sig_order=4)
    >>> pipeline = DataPipeline(data_config)
    >>> train_ds, val_ds = pipeline.prepare_data()
    >>>
    >>> # Train CVAE model
    >>> model_config = CVAEConfig(n_latent=8, n_hidden=50)
    >>> training_config = TrainingConfig(n_epochs=10000)
    >>> trainer = Trainer(model_config, training_config)
    >>> model = trainer.fit(train_ds, val_ds)
    >>>
    >>> # Generate synthetic paths
    >>> generator = MarketGenerator(model, pipeline)
    >>> generated_paths = generator.generate(n_samples=100)
"""

__version__ = '0.1.0'
__author__ = 'union'
__license__ = 'MIT'

# Import configuration classes
from .config import (
    DataPipelineConfig,
    CVAEConfig,
    TransformerConfig,
    DiffusionConfig,
    ModelConfig,
    TrainingConfig,
)

# Import base classes
from .models.base import BaseGenerativeModel
from .data.loaders.base import BaseDataLoader

# Import utilities
from .utils import (
    get_device,
    set_seed,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Version info
    '__version__',
    # Configuration
    'DataPipelineConfig',
    'CVAEConfig',
    'TransformerConfig',
    'DiffusionConfig',
    'ModelConfig',
    'TrainingConfig',
    # Base classes
    'BaseGenerativeModel',
    'BaseDataLoader',
    # Utilities
    'get_device',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
]
