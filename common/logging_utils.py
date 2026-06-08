# logging_utils.py
import os
import logging
from common import config  # local config
from contextlib import contextmanager

@contextmanager
def per_sample_log(sample_id, dataset, base_dir="log"):
    import os, logging, datetime
    """Write to log/<dataset>/<sample_id>.log only within the with-block."""
    os.makedirs(f"{base_dir}/{dataset}", exist_ok=True)
    logger = logging.getLogger()

    log_path = f"{base_dir}/{dataset}/{sample_id}{config.ADDITIONAL_RE}.log"
    fh = logging.FileHandler(log_path, mode='a', encoding="utf-8", delay=True)  # append
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    logger.addHandler(fh)
    try:
        logger.info("---- RUN START %s ----", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        yield
    finally:
        logger.removeHandler(fh)
        fh.close()
