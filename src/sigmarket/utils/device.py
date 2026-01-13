"""Device management utilities."""

import torch
from typing import Union, Optional


def get_device(device: Union[str, torch.device] = 'auto') -> torch.device:
    """Get PyTorch device for computation.

    Args:
        device: Device specification
            - 'auto': Automatically select CUDA if available, else CPU
            - 'cuda' or 'cuda:0', 'cuda:1', etc.: Specific CUDA device
            - 'cpu': CPU device
            - torch.device object: Use as-is

    Returns:
        torch.device object

    Raises:
        ValueError: If device string is invalid
        RuntimeError: If CUDA requested but not available
    """
    if isinstance(device, torch.device):
        return device

    device_str = str(device).lower().strip()

    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    elif device_str == 'cuda' or device_str.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but CUDA is not available. "
                "Please check your PyTorch installation and GPU drivers."
            )
        return torch.device(device_str)

    elif device_str == 'cpu':
        return torch.device('cpu')

    else:
        raise ValueError(
            f"Invalid device '{device}'. Must be 'auto', 'cuda', 'cuda:N', 'cpu', or torch.device"
        )


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """Move data to specified device.

    Handles tensors, dictionaries, lists, and tuples recursively.

    Args:
        data: Data to move (tensor, dict, list, or tuple)
        device: Target device

    Returns:
        Data moved to device (same structure as input)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)

    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}

    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]

    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)

    else:
        # Return as-is for non-tensor types (int, float, str, etc.)
        return data


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device information:
            - 'cuda_available': bool
            - 'cuda_device_count': int
            - 'cuda_device_names': list of str
            - 'current_device': int (if CUDA available)
            - 'cuda_version': str (if CUDA available)
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_device_names': [],
    }

    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
        info['current_device'] = torch.cuda.current_device()
        info['cuda_version'] = torch.version.cuda

    return info


def set_device(device: Union[str, int, torch.device]) -> None:
    """Set current CUDA device.

    Args:
        device: Device to set as current
            - int: CUDA device index
            - str: 'cuda:N' where N is device index
            - torch.device: CUDA device object

    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If device specification is invalid
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if isinstance(device, int):
        torch.cuda.set_device(device)
    elif isinstance(device, str):
        if device.startswith('cuda:'):
            device_id = int(device.split(':')[1])
            torch.cuda.set_device(device_id)
        else:
            raise ValueError(f"Invalid device string: {device}")
    elif isinstance(device, torch.device):
        if device.type == 'cuda':
            torch.cuda.set_device(device.index if device.index is not None else 0)
        else:
            raise ValueError(f"Device must be CUDA device, got {device}")
    else:
        raise ValueError(f"Invalid device type: {type(device)}")


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_info() -> dict:
    """Get GPU memory information.

    Returns:
        Dictionary with memory info (empty if CUDA not available):
            - 'allocated': Currently allocated memory (bytes)
            - 'reserved': Currently reserved memory (bytes)
            - 'max_allocated': Maximum allocated memory (bytes)
            - 'max_reserved': Maximum reserved memory (bytes)
    """
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated': torch.cuda.memory_allocated(),
        'reserved': torch.cuda.memory_reserved(),
        'max_allocated': torch.cuda.max_memory_allocated(),
        'max_reserved': torch.cuda.max_memory_reserved(),
    }


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()

    print("="  * 60)
    print("PyTorch Device Information")
    print("=" * 60)
    print(f"CUDA Available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Number of CUDA Devices: {info['cuda_device_count']}")
        print(f"Current CUDA Device: {info['current_device']}")
        print("\nCUDA Devices:")
        for i, name in enumerate(info['cuda_device_names']):
            print(f"  [{i}] {name}")

        # Memory info
        mem_info = get_memory_info()
        print(f"\nCurrent Device Memory:")
        print(f"  Allocated: {mem_info['allocated'] / 1024**2:.2f} MB")
        print(f"  Reserved: {mem_info['reserved'] / 1024**2:.2f} MB")
    else:
        print("CUDA not available - using CPU")

    print("=" * 60)
