import logging
import os
from logging import handlers


class LoggerSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Call setup_logging with default values
        self.setup_logging(
            log_dir="logs",
            log_file="app.log",
            log_level_str="INFO",
            max_bytes=5 * 1024 * 1024,  # 5 MB
            backup_count=5,
        )
        self._initialized = True

    def setup_logging(
        self,
        log_dir: str,
        log_file: str,
        log_level_str: str,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """
        Sets up logging based on the loaded configuration.
        """

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Define the full path to the log file
        log_path = os.path.join(log_dir, log_file)

        # Remove all existing handlers associated with the root logger
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Set the root logger level
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        root_logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create File Handler with Rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,  # 5 MB
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Create Stream Handler for stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Add handlers to the root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

        root_logger.info(f"Logging initialized. Logs are being saved to {log_path}")


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    LoggerSingleton()  # Initialize logging
    return logging.getLogger(name)


if __name__ == "__main__":
    # Initialize the logger
    LoggerSingleton()

    # Get a named logger instance
    logger = get_logger(__name__)
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
