###
# Logger module for the RAG pipeline
# provides logging for all levels
# takes the caller information and log as kwargs, and writes them to log file
###

import logging
import sys
import os

from datetime import datetime

class ImmediateFileHandler(logging.FileHandler):
    """FileHandler that forces an OS-level flush to disk on every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            if self.stream is not None:
                os.fsync(self.stream.fileno())
        except OSError:
            pass

LOG_FILE = f"logs/rag_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logger(name: str = "rag_agent", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a dedicated logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times if logger is requested again
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Optional: Add file handler if needed
        os.makedirs("logs", exist_ok=True)
        file_handler = ImmediateFileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Default logger instance for direct imports
logger = setup_logger()

def log_info(message: str, **kwargs):
    logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    logger.error(message, **kwargs)

def log_warning(message: str, **kwargs):
    logger.warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    logger.debug(message, **kwargs)

def log_critical(message: str, **kwargs):
    logger.critical(message, **kwargs)