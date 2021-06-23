import time

__all__ = ['Benchmark']

class Benchmark:
    def __init__(self, enabled: bool = True, logger = None) -> None:

        self._enabled = enabled
        self._start = None
        self._logger = logger

        self.start()

    def start(self):
        if self._enabled:
            self._start = time.time()

    def elapsed(self):
        if self._enabled:
            msg = f'Elapsed: {time.time() - self._start:>.3f}s'
            if self._logger:
                self._logger.info(msg)
            else:
                print(msg)

