# SigMarket: Signature-Based Market Simulation Library

A modern PyTorch-based library for generating synthetic financial market data using signature methods and deep generative models.

## Overview

SigMarket implements the signature-based approach to market generation described in:

> H. Buehler, B. Horvath, T. Lyons, I. Perez Arribas and B. Wood.
> **"Generating financial markets with signatures."**
> SSRN 3657366, 2020.

The library provides a clean, modular architecture with three distinct pipelines:

1. **Data Pipeline**: Load, preprocess, compute signatures
2. **Training Pipeline**: Train generative models (CVAE, Transformer, Diffusion)
3. **Generation Pipeline**: Generate synthetic paths and validate

## Key Features

- **GPU-Accelerated Signatures**: Uses [Signatory](https://github.com/patrick-kidger/signatory) for 10-100x speedup
- **Multiple Generative Models**: CVAE, Transformer, and Diffusion models
- **Multi-Asset Support**: Handle correlated multi-dimensional time series
- **Flexible Data Sources**: Yahoo Finance, CSV, Parquet, synthetic (Rough Bergomi)
- **Advanced Preprocessing**: Log returns, volume, volatility, time-of-day features
- **Modern PyTorch**: Clean, type-hinted code with comprehensive documentation
- **Extensible Architecture**: Easy to add new models, data sources, and features

## Installation

### From Source

```bash
git clone https://github.com/union/sigmarket.git
cd Signature_Market_Sim
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev,viz,notebooks]"
```

## Quick Start

### Single-Asset Market Generation

```python
from sigmarket import DataPipeline, Trainer, MarketGenerator
from sigmarket.config import DataPipelineConfig, CVAEConfig, TrainingConfig

# 1. Configure and prepare data
data_config = DataPipelineConfig(
    loader_type='yahoo',
    ticker='AAPL',
    start='2010-01-01',
    end='2020-01-01',
    freq='M',
    sig_order=4,
    transforms=['leadlag']
)

pipeline = DataPipeline(data_config)
train_ds, val_ds = pipeline.prepare_data()

# 2. Train CVAE model
model_config = CVAEConfig(n_latent=8, n_hidden=50, alpha=0.003)
training_config = TrainingConfig(
    n_epochs=10000,
    learning_rate=0.005,
    device='cuda'
)

trainer = Trainer(model_config, training_config)
model = trainer.fit(train_ds, val_ds)

# 3. Generate synthetic paths
generator = MarketGenerator(model, pipeline)
generated_paths = generator.generate(
    n_samples=100,
    condition=train_ds[-1]['logsig'],
    validate=True
)
```

## Project Structure

```
Signature_Market_Sim/
├── src/sigmarket/       # New PyTorch implementation
│   ├── config/          # Dataclass-based configurations
│   ├── data/            # Data pipeline
│   ├── models/          # Generative models (CVAE, Transformer, Diffusion)
│   ├── training/        # Training pipeline
│   ├── generation/      # Generation pipeline
│   ├── utils/           # Utilities
│   └── compat/          # Backward compatibility
└── market_simulator/    # Original TensorFlow implementation (reference)
```

## Development Status

**Phase 1: Core Infrastructure** ✅ Complete
- [x] Package structure setup
- [x] Configuration system (dataclasses)
- [x] Base classes and interfaces
- [x] Utility modules (device, I/O, seed)

**Phase 2: Data Pipeline** (In Progress)
- [ ] Data loaders (Yahoo, CSV, synthetic)
- [ ] Lead-lag transformation
- [ ] Signatory integration
- [ ] Normalization and datasets

**Phase 3: CVAE Model** (Planned)
- [ ] Port CVAE from TensorFlow to PyTorch
- [ ] Encoder/decoder architecture
- [ ] VAE loss implementation
- [ ] Validate against original

**Phase 4: Training Pipeline** (Planned)
- [ ] Trainer class implementation
- [ ] Callbacks system
- [ ] End-to-end training test

**Phase 5: Generation Pipeline** (Planned)
- [ ] Generation logic
- [ ] Port genetic algorithm inversion
- [ ] Validation integration

**Phase 6+: Advanced Models** (Planned)
- [ ] Transformer architecture
- [ ] Diffusion model
- [ ] Multi-asset support
- [ ] Documentation and examples

## Citation

If you use this library, please cite the original paper:

```bibtex
@article{buehler2020generating,
  title={Generating financial markets with signatures},
  author={Buehler, Hans and Horvath, Blanka and Lyons, Terry and Perez Arribas, Imanol and Wood, Ben},
  journal={Available at SSRN 3657366},
  year={2020}
}
```

## License

MIT License - see LICENSE file for details.