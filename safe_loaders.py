import argparse
import logging
import torch


logger = logging.getLogger(__name__)


def safe_torch_load_args(path: str):
    """
    PyTorch 2.6+ defaults to weights_only=True and blocks unpickling argparse.Namespace.
    This allowlists argparse.Namespace safely (assuming you trust the checkpoint).
    """
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path)
