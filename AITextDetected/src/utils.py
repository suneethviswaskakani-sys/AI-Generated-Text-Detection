"""Shared utilities: logging setup and label normalization."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_logger(name: str) -> logging.Logger:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(config.LOGS_DIR, "app.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def normalize_label(x) -> int:
    """Map varied label representations to {0, 1}. 1 = AI-generated."""
    s = str(x).strip().upper()
    if s in config.AI_LABELS:
        return 1
    if s in config.HUMAN_LABELS:
        return 0
    if "AI" in s:
        return 1
    return 0
