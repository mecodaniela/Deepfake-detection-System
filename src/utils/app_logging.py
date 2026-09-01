"""
logging.py — Standardizon log-et në gjithë projektin. Format i thjeshtë,
konsistent: [NIVELI] mesazhi, me timestamp opsional.
"""

import logging
import sys


def get_logger(name: str = "deepfake_detection", level: int = logging.INFO) -> logging.Logger:
    """
    Kthen një logger të konfiguruar njësoj kudo në projekt — thirre
    një herë në krye të çdo skripti: `log = get_logger(__name__)`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:  # shmang dublikim handler-ësh nëse thirret disa herë
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Loading image")
    log.info("SHA-256 calculated")
    log.warning("This is a warning example")
    log.error("This is an error example")