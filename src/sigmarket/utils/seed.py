"""Utilities for reproducibility via random seed setting."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    This sets the seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make CuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_for_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Set random seed with optional deterministic mode.

    Args:
        seed: Random seed value
        deterministic: If True, enforces fully deterministic behavior (slower)
                      If False, allows some non-deterministic operations (faster)
    """
    set_seed(seed)

    if deterministic:
        # Full determinism
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Allow non-deterministic but faster operations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_state() -> dict:
    """Get current random state from all libraries.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: dict) -> None:
    """Restore random state for all libraries.

    Args:
        state: Dictionary containing random states (from get_random_state())
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
