import time
from json import dumps
import torch
import torchvision
import platform

__all__ = ['Benchmark', 'env_info']

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

def env_info(json: bool = False):
    kvs = {
        'Python': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'CUDA': torch.version.cuda,
        'cuDNN': torch.backends.cudnn.version(),
        'GPU': {
           f'#{i}' : { 
               'name': torch.cuda.get_device_name(i), 
               'memory': f'{torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024):.2f}GB' 
            } 
            for i in range(torch.cuda.device_count() )
        },
        'Platform': {
            'system': platform.system(),
            'node': platform.node(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    }

    return kvs if not json else dumps(kvs, indent=4, separators=(',', ':'))