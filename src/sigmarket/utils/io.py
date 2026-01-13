"""I/O utilities for model and data persistence."""

import torch
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save (optional)
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
        additional_info: Additional information to save (e.g., scheduler state, metrics)
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if additional_info is not None:
        checkpoint['additional_info'] = additional_info

    torch.save(checkpoint, save_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint information (epoch, loss, additional_info)
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'additional_info': checkpoint.get('additional_info', {}),
    }


def save_model(
    model: torch.nn.Module,
    path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save model weights and configuration.

    Args:
        model: Model to save
        path: Path to save model
        config: Optional configuration dictionary
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if config is not None:
        save_dict['config'] = config

    torch.save(save_dict, save_path)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: str = 'cpu'
) -> Optional[Dict[str, Any]]:
    """Load model weights.

    Args:
        model: Model to load weights into
        path: Path to saved model
        device: Device to load tensors to

    Returns:
        Configuration dictionary if saved, None otherwise
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint.get('config')


def save_pickle(obj: Any, path: str) -> None:
    """Save object using pickle.

    Args:
        obj: Object to save
        path: Path to save file
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Load object using pickle.

    Args:
        path: Path to pickle file

    Returns:
        Loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save dictionary as JSON.

    Args:
        data: Dictionary to save
        path: Path to save file
        indent: Indentation for pretty printing
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path).expanduser()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_checkpoints(checkpoint_dir: str, pattern: str = '*.pt') -> list:
    """List all checkpoint files in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match (default: '*.pt')

    Returns:
        List of checkpoint file paths (sorted by modification time)
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return []

    checkpoints = sorted(
        checkpoint_path.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return [str(p) for p in checkpoints]


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = '*.pt') -> Optional[str]:
    """Get the most recent checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match (default: '*.pt')

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    return checkpoints[0] if checkpoints else None
