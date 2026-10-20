"""Pick the best available compute device, so the same code runs on an NVIDIA GPU, a Mac or a CPU."""

import torch


def get_device() -> str:
    """Return a device string Ultralytics understands.

    - "0"   -> first NVIDIA GPU
    - "mps" -> Apple Silicon GPU
    - "cpu" -> fallback
    """
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
