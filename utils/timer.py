from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from utils.console import Timer

timer = Timer(disabled=False, sync_cuda=False)  # when initializing, always disable


def call_with_timeout(fn, timeout, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)


def timeout(seconds: float, *, timeout_exception=TimeoutError):
    """
    Soft timeout decorator.
    - Does NOT kill the function, only stops waiting.
    - Works cross-platform.
    - Best for Python / I/O-bound code.

    Args:
        seconds: timeout in seconds (float)
        timeout_exception: exception to raise on timeout
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    raise timeout_exception(
                        f"{fn.__name__} timed out after {seconds}s"
                    )
        return wrapper
    return decorator


def log_timing(name: str):
    from configs import debug_options
    if not debug_options.DEBUG:
        return

    import torch
    from utils.distributed import is_node_main

    if is_node_main():
        timer.record(f'{torch.cuda.memory_allocated() / 2**30:.2f} GB {name}')
