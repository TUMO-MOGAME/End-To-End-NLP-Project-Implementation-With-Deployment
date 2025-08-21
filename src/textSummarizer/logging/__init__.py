import os
import sys
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("textSummarizer")

def get_log_level() -> int:
    return logging.DEBUG if os.environ.get("DEBUG") else logging.INFO

def setup_logger(log_file="app.log") -> None:
    if logger.hasHandlers():
        return  # prevent duplicate handlers

    logger.setLevel(get_log_level())

    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] Message: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logger initialized successfully")  # keep ✅ only if UTF-8 safe console
