"""Configuration Utilities"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from omegaconf import OmegaConf

from sat.utils import logging

logger = logging.get_default_logger("sat.utils.config")


def _cuts_max(path: str) -> float:
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        logger.warning(f"cuts_max: {path} not found, using a time scale of 1.0")
        return 1.0
    with p.open() as f:
        values = [float(line) for line in f if line.strip()]
    return values[-1]


class Config(object):
    """Configure the omega configuration environment."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for the configuration class."""
        if cls._instance is None:
            logger.debug("Create new Config object")
            cls._instance = super(Config, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        """Add new resolvers to OmegaConf."""
        logger.debug("Initialize Config object")
        OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
        OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        # ${cuts_max:<path to duration_cuts.csv>} -> the last cut, i.e. the end of
        # follow-up; the default time unit of the regression head and of MMV (v2)
        OmegaConf.register_new_resolver("cuts_max", _cuts_max)
