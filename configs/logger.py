import os
import logging
import structlog
from logging.handlers import TimedRotatingFileHandler


class CustomLogger:
    _configured = False
    _loggers = {}

    def __init__(self, log_dir="logs", log_file="app.log"):
        """
        log_dir: directory to store logs
        log_file: base file name (rotates daily)
        """
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def _configure_logging(self):
        if CustomLogger._configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        if not root_logger.handlers:
            # Rotating file handler (daily, keep 30 backups)
            file_handler = TimedRotatingFileHandler(
                self.log_file_path, when="midnight", backupCount=30, utc=True
            )
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,  # important for async/context
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        CustomLogger._configured = True

    def get_logger(self, name="root"):
        """
        Returns a cached structured logger
        """
        self._configure_logging()
        if name not in self._loggers:
            self._loggers[name] = structlog.get_logger(name)
        return self._loggers[name]


GLOBAL_LOGGER = CustomLogger().get_logger("app")
