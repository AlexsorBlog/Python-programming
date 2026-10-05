import functools
import logging
import time

log = logging.getLogger("findex.timing")


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug(
                "%s -> %.2f мс", fn.__name__, (time.perf_counter() - start) * 1000
            )

    return wrapper
