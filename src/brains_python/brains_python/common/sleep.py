from time import perf_counter

__all__ = ["sleep"]


def sleep(duration):
    start = perf_counter()
    while perf_counter() - start < duration:
        pass
