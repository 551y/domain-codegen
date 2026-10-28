import logging, sys

CONVERSION_LOG = logging.getLogger("conversion")
TRACER_LOG = logging.getLogger("trace_semantic")
VALIDATION_LOG = logging.getLogger("validation")

TF_LOG = logging.getLogger("gen|tf")
PADDLE_LOG = logging.getLogger("gen|paddle")
TORCH_LOG = logging.getLogger("gen|torch")
MS_LOG = logging.getLogger("gen|ms")

ALL_LOGGERS = [CONVERSION_LOG, TF_LOG, PADDLE_LOG, TORCH_LOG, MS_LOG]


def setup_run_logging(log_file_path, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    for logger in ALL_LOGGERS:
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(level)
        logger.propagate = False

    logging.getLogger().addHandler(file_handler)