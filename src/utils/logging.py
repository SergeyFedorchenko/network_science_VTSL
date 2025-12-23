"""
Centralized logging configuration.

Provides console and file logging with consistent formatting.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Configure a logger with console and/or file output.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Optional path to log file (under results/logs/)
        level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """
    Setup root logger for a script.
    
    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )


def get_script_logger(script_name: str, results_dir: Path) -> logging.Logger:
    """
    Get a logger for a script with automatic file path under results/logs/.
    
    Args:
        script_name: Name of the script (e.g., "00_validate_inputs")
        results_dir: Path to results directory
        
    Returns:
        Configured logger
    """
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"
    
    return setup_logger(
        name=script_name,
        log_file=log_file,
        level=logging.INFO,
        console=True
    )
