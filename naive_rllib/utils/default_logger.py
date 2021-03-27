from loguru import logger
import sys

__all__ = ['get_logger']


def get_logger(name: str = None, level: str = "DEBUG", rotation: str = '500 MB', retention: str = '3 days'):
    sink = sys.stderr if name is None else name
    configs = dict(sink=sink, format="{time:YYYY-MM-DD HH:mm:ss:SSSS} {level} {module}.py: {message}", level=level)
    if name is not None:
        configs.update(dict(rotation=rotation, retention=retention, encoding="utf-8"))
    logger.add(**configs)
    return logger