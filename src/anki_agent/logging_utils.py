import logging


def _parse_level(level_str: str | None) -> int:
    if not level_str:
        return logging.INFO
    level_str = level_str.strip().upper()
    # Allow numeric levels or names like DEBUG/INFO/WARNING/ERROR/CRITICAL
    if level_str.isdigit():
        return int(level_str)
    return getattr(logging, level_str, logging.INFO)


class LogsHandler:
    def __init__(self, name: str = None):
        self.log_level = logging.INFO
        self._logger_name = name if name else "anki_agent"

    def setup_logging(self, level: str = None):
        if isinstance(level, int):
            resolved = level
        elif isinstance(level, str):
            resolved = _parse_level(level)
        else:
            resolved = self.log_level
        self.log_level = resolved

        # configure only our logger
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(resolved)
        print(f"log level set to {level}")

        if not logger.handlers:  # avoid adding duplicates
            handler = logging.StreamHandler()
            handler.setLevel(resolved)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            print("added handler to logger")

    def set_log_level(self, level: str | int):
        if isinstance(level, int):
            resolved = level
        elif isinstance(level, str):
            resolved = _parse_level(level)
        else:
            raise ValueError(f"Invalid log level: {level}")
        self.log_level = resolved
        self._apply_level(resolved)

    def _apply_level(self, level: int):
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)

    def get_logger(self, name: str = None) -> logging.Logger:
        """Return a module logger (does not configure handlers)."""
        logger = logging.getLogger(name if name else "anki_agent")
        print(f"LOG LEVEL {self.log_level}")
        logger.setLevel(self.log_level)  ## making sure
        return logger


logs_handler = LogsHandler()
