import sys

from . import logger  # noqa: F401
from .data import DataModule  # noqa: F401
from .gan import GAN  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
