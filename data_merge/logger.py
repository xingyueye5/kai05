import logging
from logging import handlers
import io
from pathlib import Path
import multiprocessing as mp

base_dir = Path(__file__).resolve().parent
LOG_FILE = base_dir / 'logs/' / str(Path(__file__).with_suffix('.log').name)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = "%(asctime)s [%(processName)s] %(levelname)s %(message)s"


def setup_logging(log_file: Path = LOG_FILE, log_queue=None):
    """Configure logging for a process.

    If log_queue is provided, logs are sent to QueueListener in the main process.
    Otherwise, fall back to local file handler (single-process use).
    """
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.INFO)

    if log_queue is not None:
        queue_handler = handlers.QueueHandler(log_queue)
        root.addHandler(queue_handler)
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)


def start_logging_listener(log_file: Path = LOG_FILE, console: bool = True):
    """Start a QueueListener in the main process for multi-process safe logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    try:
        log_file.chmod(0o644)
    except Exception:
        # best-effort; ignore if filesystem does not support chmod
        pass
    log_queue = mp.Queue(-1)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = handlers.WatchedFileHandler(log_file, mode='a', encoding='utf-8', delay=False)
    file_handler.setFormatter(formatter)

    handlers_list = [file_handler]
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers_list.append(console_handler)

    listener = handlers.QueueListener(log_queue, *handlers_list)
    listener.start()
    return log_queue, listener, handlers_list


class TqdmToLogger(io.StringIO):
    """Redirect tqdm output to logger."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        buf = buf.rstrip()
        if buf:
            self.logger.log(self.level, buf)

    def flush(self):
        pass