import logging
from logging import Logger
from typing import Mapping
from config.config import load_config

cfg = load_config("logging")

def setup_logging(cfg: Mapping | None = None) -> None:
    cfg = cfg or {}
    level = getattr(logging, cfg.get("level", "INFO"))
    fmt = cfg.get("format", "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s")
    datefmt = cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

def get_logger(name: str) -> Logger:
    return logging.getLogger(name)