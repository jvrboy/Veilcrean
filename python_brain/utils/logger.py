"""
logger.py
=========
Centralized logging for the Python brain. Files rotate daily.
"""
from __future__ import annotations
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from ..config import LOGS_DIR


_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str,
               level: int = logging.INFO,
               to_file: bool = True,
               log_dir: Optional[Path] = None) -> logging.Logger:
    """Build (or fetch) a named logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(_FMT)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File (daily rotation)
    if to_file:
        d = Path(log_dir or LOGS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            d / "veilcrean.log", when="midnight", backupCount=14, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
