"""
Lightweight colored logger helper.

- Uses `rich` (if available) for nicer console output.
- Falls back to ANSI-colored formatter when `rich` is unavailable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


class _AnsiColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[36m",  # cyan
        logging.INFO: "\x1b[32m",  # green
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",  # red
        logging.CRITICAL: "\x1b[35m",  # magenta
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        if not color:
            return base
        return f"{color}{base}{self.RESET}"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create/get a logger with colored console output.

    Args:
        name: logger name.
        level: logging level.
        log_file: optional path to log file (plain text).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid double handlers if called multiple times
    if getattr(logger, "_morl_configured", False):
        return logger

    logger.propagate = False

    # File handler (optional)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(fh)

    # Console handler
    try:
        from rich.logging import RichHandler  # type: ignore

        ch = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            log_time_format="%H:%M:%S",
        )
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    except Exception:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(
            _AnsiColorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(ch)

    logger._morl_configured = True  # type: ignore[attr-defined]
    return logger

