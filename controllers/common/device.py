"""PyTorch device selection for Apple Silicon."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def select_torch_device(requested: str = "auto") -> torch.device:
    """Select MPS when available, otherwise use the CPU."""
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError(f"Unsupported device {requested!r}; choose auto, mps, or cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if requested == "mps":
        warnings.warn("MPS is unavailable; falling back to CPU", RuntimeWarning, stacklevel=2)
    return torch.device("cpu")


def configure_torch(seed: int, threads: int, deterministic: bool = True) -> None:
    """Configure repeatable PyTorch execution without accelerator-specific calls."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
