import sys

from loguru import logger as log

__all__ = ["log"]


def formatter(record):
    """small loguru modification to allow multi-line logging"""

    lines = record["message"].splitlines()
    prefix = (
        "{time:YY-MM-DD HH:mm:ss.S} | {level.name:<8} | "
        + "{file}.{function}:{line} - ".format(**record)
    )
    indented = (
        lines[0] + "\n" + "\n".join(" " * len(prefix) + line for line in lines[1:])
    )
    record["message"] = indented.strip()
    return (
        "<g>{time:YY-MM-DD HH:mm:ss.S}</> | <lvl>{level.name:<8}</> | "
        + "<e>{file}.{function}:{line}</> - <lvl>{message}\n</>{exception}"
    )


log.remove()
log.add(sys.stderr, format=formatter)
