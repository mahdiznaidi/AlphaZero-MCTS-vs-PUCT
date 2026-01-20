from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO, filepath: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler: logging.Handler
    if filepath:
        handler = logging.FileHandler(filepath)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
