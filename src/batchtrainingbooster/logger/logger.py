"""Logger module for the package"""

from logging import getLogger, Formatter, INFO, StreamHandler
from logging.handlers import RotatingFileHandler
from os import makedirs, path

ROOT_DIR_LOGS = path.join(path.dirname(__file__), "logs")

if not path.exists(ROOT_DIR_LOGS):
    makedirs(ROOT_DIR_LOGS)


def setup_logger(name: str, log_path: str = ROOT_DIR_LOGS, level: int = INFO):
    """
    Setup a logger for the application.

    Args:
        name (str): The name of the logger.
        log_path (str): The directory path where log files will be stored.
        level (int): The logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).


    """

    if len(path.split(name)) > 1:
        name = path.split(name)[-1]

    logger = getLogger(name)

    if not logger.handlers:
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # create the log folder
        makedirs(log_path, exist_ok=True)
        log_file = path.join(log_path, f"{name}.log")

        # Console handler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,  # 5 MB
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Console handler
        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    logger.propagate = False
    logger.setLevel(level)

    return logger
