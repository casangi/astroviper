import threading

import numpy as np


def _linear_algebra(seed):
    """Products and a solve of the kinds node tasks run, on inputs of their own."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(4128, 136))
    products = rng.normal(size=(136, 4))
    coefficients = rng.normal(size=(4128, 2))
    basis = rng.normal(size=(166, 2))
    image = rng.normal(size=(524288, 4)).astype(np.float32)
    mixing = rng.normal(size=(4, 4)).astype(np.float32)
    square = rng.normal(size=(300, 300))
    normal = square @ square.T + 300 * np.eye(300)
    return [
        lambda: weights @ products,
        lambda: coefficients @ basis.T,
        lambda: np.tensordot(image, mixing, ((1,), (1,))),
        lambda: np.linalg.solve(normal, square[:, :8]),
    ]


def test_threads_of_one_process_get_the_results_of_one_thread():
    """tests/conftest.py limits BLAS to one thread: OpenBLAS before 0.3.34 gives
    wrong results when threads of one process call it at once, as the node
    tasks of Dask's threaded scheduler do."""
    n_threads = 3
    calls = [_linear_algebra(seed) for seed in range(n_threads)]
    expected = [[call() for call in thread_calls] for thread_calls in calls]
    barrier = threading.Barrier(n_threads)
    wrong = []

    def work(index):
        # Each thread starts with another call, so different kinds overlap.
        order = np.roll(np.arange(len(calls[index])), -index)
        try:
            for _ in range(30):
                barrier.wait()
                for k in order:
                    if not np.array_equal(calls[index][k](), expected[index][k]):
                        wrong.append((index, int(k)))
        except Exception as error:  # a corrupted factorization can raise
            wrong.append((index, repr(error)))
            barrier.abort()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wrong == []
