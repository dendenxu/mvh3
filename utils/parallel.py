import math
import time
from typing import Callable, Dict, List
from multiprocessing.pool import Pool, ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import Future

from utils.console import dotdict
from utils.console import tqdm
from utils.console import tqdm_rich
from utils.console import log
from utils.timer import timer


def read_process_write(n_total: int,
                       chunk_size: int,
                       read: Callable,
                       process: Callable,
                       write: Callable,
                       reuse_executor: bool = False,  # whether to reuse the task executor for better coordination
                       use_process: bool = False,
                       buffer_size: int = 1,
                       num_workers=8,
                       pass_idx_to_process: bool = False,
                       verbose: bool = True,
                       desc: str = 'read_process_write',
                       start_offset: int = 0,
                       print_progress: bool = False,
                       ):
    if use_process:
        executor = ProcessPoolExecutor(num_workers)
    else:
        executor = ThreadPoolExecutor(num_workers)
    next_future = None
    save_future = None
    n_chunks = math.ceil(n_total / chunk_size)

    next_futures = []
    save_futures = []

    t0 = time.time()
    chunk_iter = tqdm(range(n_chunks), desc=desc, disable=not print_progress)
    for chunk_idx in chunk_iter:
        start_idx = chunk_idx * chunk_size + start_offset
        end_idx = min((chunk_idx + 1) * chunk_size + start_offset, n_total)

        # minimal ETA print (average chunk time so far)
        if verbose:
            if chunk_idx > 0:
                elapsed = time.time() - t0
                avg = elapsed / chunk_idx
                remaining = avg * (n_chunks - chunk_idx)
                eta_sec = max(0, int(remaining))
                hh = eta_sec // 3600
                mm = (eta_sec % 3600) // 60
                ss = eta_sec % 60
                eta = f' ETA {hh:02d}:{mm:02d}:{ss:02d}'
            else:
                eta = ' ETA N/A'
            log(f'{desc}: chunk {chunk_idx + 1}/{n_chunks} (index range {start_idx}-{end_idx - 1}){eta}')

        # If we have pre-loaded futures from previous iteration, get their results
        if len(next_futures):
            if verbose:
                log(f'Getting pre-loaded chunk results...')
            loaded = next_futures.pop(0).result()
        else:
            # First chunk needs to be loaded synchronously
            if verbose:
                log(f'Loading first chunk of {chunk_size} data blocks to CUDA...')
            if reuse_executor:
                loaded = read(start_idx, end_idx, executor)
                loaded = loaded.result()
            else:
                loaded = read(start_idx, end_idx)

        # Start loading next chunk in background
        while len(next_futures) <= buffer_size:
            next_chunk_idx = chunk_idx + len(next_futures) + 1  # next chunk to load, 0-indexed
            if next_chunk_idx < n_chunks - 1:
                if verbose:
                    log(f'Starting background load of chunk {next_chunk_idx + 1}...')
                next_start_idx = next_chunk_idx * chunk_size
                next_end_idx = min((next_chunk_idx + 1) * chunk_size, n_total)
                if reuse_executor:
                    next_future = read(next_start_idx, next_end_idx, executor)
                else:
                    next_future = executor.submit(read, next_start_idx, next_end_idx)  # parallelize yourself in the read function
                next_futures.append(next_future)
            else:
                break

        # Process current chunk
        if verbose:
            log(f'Running the process function...')
        if pass_idx_to_process:
            output = process(loaded, start_idx, end_idx)
        else:
            output = process(loaded)

        # Wait for previous chunk's saving to complete if any
        if len(save_futures) == buffer_size:
            if verbose:
                log('Waiting for buffered chunk save to complete...')
            save_futures.pop(0).result()

        # Start saving results in background
        if verbose:
            log('Starting background save of results...')
        if reuse_executor:
            save_future = write(output, start_idx, end_idx, executor)
        else:
            save_future = executor.submit(write, output, start_idx, end_idx)
        save_futures.append(save_future)

    # Wait for final saves to complete
    while len(save_futures):
        if verbose:
            log('Waiting for buffered chunk save to complete...')
        save_futures.pop(0).result()

    executor.shutdown()


def parallel_execution(  # noqa: C901
    *args,
    action: Callable,
    num_workers=32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    use_process=False,
    callback=lambda x, y, z: z,  # return results directly
    force_keep_kwargs=(),
    **kwargs,
):
    """
    Executes a given function in parallel using threads or processes.
    When using threads, the parallelism is achieved during IO blocking (i.e. when loading images from disk or writing something to disk).
    If your task is compute intensive, consider using packages like numpy or torch since they release the GIL during heavy lifting.

    Args:
        *args: Variable length argument list.
        action (Callable): The function to execute in parallel.
        num_workers (int): The number of worker threads or processes to use.
        print_progress (bool): Whether to print a progress bar.
        sequential (bool): Whether to execute the function sequentially instead of in parallel.
        async_return (bool): Whether to return a pool object for asynchronous results.
        desc (str): The description to use for the progress bar.
        use_process (bool): Whether to use processes instead of threads.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        If `async_return` is False, returns a list of the results of executing the function on each input argument.
        If `async_return` is True, returns a pool object for asynchronous results.
    """

    # https://superfastpython.com/threadpool-python/
    # Python threads are well suited for use with IO-bound tasks
    # Note: we expect first arg / or kwargs to be distributed

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [
            (arg[i] if isinstance(arg, list) and len(arg) == length else arg)
            for arg in args
        ]
        # TODO: Support all types of iterable
        action_kwargs = {
            key: (
                kwargs[key][i]
                if isinstance(kwargs[key], list)
                and len(kwargs[key]) == length
                and key not in force_keep_kwargs
                else kwargs[key]
            )
            for key in kwargs
        }
        return action_args, action_kwargs

    if issubclass(tqdm, tqdm_rich):
        tqdm_kwargs = dotdict(back=3, desc=desc, disable=not print_progress)
    else:
        tqdm_kwargs = dotdict(desc=desc, disable=not print_progress)

    sequential = sequential or num_workers == 0  # similar to pytorch multiprocess dataloading

    if not sequential:
        # Create ThreadPool
        if use_process:
            pool = Pool(processes=num_workers)
        else:
            pool = ThreadPool(processes=num_workers)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for i, async_result in tqdm(
                enumerate(asyncs), total=len(asyncs), **tqdm_kwargs
            ):  # log previous frame
                result = async_result.get()  # will sync the corresponding thread
                result = callback(i, len(asyncs), result)
                results.append(result)  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), **tqdm_kwargs):  # log previous frame
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            result = action(*action_args, **action_kwargs)
            result = callback(i, length, result)
            results.append(result)
        return results
