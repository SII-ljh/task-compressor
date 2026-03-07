"""Task Compressor — compress ultra-long texts into fixed-length latent representations."""

from .config import Config
from .models import TaskCompressorModel

__all__ = ["Config", "TaskCompressorModel"]
