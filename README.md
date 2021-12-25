# Faster Training

Single-machine multi-GPU 

## ENV
```json
{
    "Python":"3.8.10",
    "torch":"1.8.1",
    "torchvision":"0.9.1",
    "dali": "1.2.0",
    "CUDA":"11.1",
    "cuDNN":8005,
    "GPU":{
        "#0":{
            "name":"Quadro RTX 6000",
            "memory":"23.65GB"
        },
        "#1":{
            "name":"Quadro RTX 6000",
            "memory":"23.65GB"
        }
    },
    "Platform":{
        "system":"Linux",
        "node":"4029GP-TRT",
        "version":"#83~18.04.1-Ubuntu SMP Tue May 11 16:01:00 UTC 2021",
        "machine":"x86_64",
        "processor":"x86_64"
    }
}
```

## Model Running
```
Batch size: 512, conv layers: 11, epochs: 5
```
> Baseline : 276.980s

|    |         | +cudnn_benchmark | +AMP   | +cudnn_benchmark +AMP |
|:--:|  :--:   |:--:              |:--:    | :--:                  |
| DP | 163.740 | 104.807          | 74.948 | 73.862                |
|DDP | 142.497 | 102.535          | 67.095 | 72.998                |

- [x] `DP`: `torch.nn.DataParallel`
- [x] `AMP`: `torch.cuda.amp`
- [x] `DDP`: `torch.nn.parallel.DistributedDataParallel`
- [x] `cudnn_benchmark`: `torch.backends.cudnn.benchmark = True`
- [x] `pin_memory=True`
- [x] `non_blocking=True`
- [x] `optimizer.zero_grad(set_to_none=True)`


### Usage
```bash
# $1 is the epochs
./running.sh 5

# Or run the commands in the script directly.
```

## Data Loading
### Prepare
> Drop caches for i/o benchmark test.

```bash
sync

# To free pagecache:
echo 1 > /proc/sys/vm/drop_caches
# To free reclaimable slab objects (includes dentries and inodes):
echo 2 > /proc/sys/vm/drop_caches
#To free slab objects and pagecache:
echo 3 > /proc/sys/vm/drop_caches
```

### Data Loading Time
```
Batch size: 256/2, workers: 8 x 2
```
|     |        | Bottleneck | +DALI/CPU  |  Bottleneck | +DALI/GPU  |  Bottleneck | 
|:--: |:--:    |:--:        |:--:        |:--:         |:--:        |:--:         |
| HDD | ~25M/s |    IO      | ~40M/s     | IO          |  ~40M/s    | IO          |
| SSD | ~230M/s|    CPU     | ~500M/s    | CPU         | ~600M/s    | IO          |

- [x] `SSD`
- [x] `DALI`: [The NVIDIA Data Loading Library](https://github.com/NVIDIA/DALI)
- [ ] `LMDB`

### Usage
```bash
# $1 is the script, $2 is the imagenet dataset path.
./loading.sh loading_faster.py '/datasets/ILSVRC2012/'

# Or run the commands in the script directly.
```

## Downscale ImageNet Dataset (for validating ideas quickly)
The average resolution of ImageNet images is `469x387`, but they are usually cropped to `256x256` or `224x224` in your image preprocessing step. *So we could speed up reading by downscaling the image size.*
Especially, the entire dataset can be loaded into memory.
```bash
# N: the max size of smaller edge
python resize_imagenet.py --src </path/to/imagenet> --dst </path/to/imagenet/resized> --max-size N
```

### Training with smaller size
As reported in [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423), you can use smaller image size when training models.
