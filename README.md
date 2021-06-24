# Faster

Single-machine multi-GPU 

## Run Time
> Baseline : 82.826s

|    |        | +cudnn_benchmark | +AMP   | +cudnn_benchmark +AMP |
|:--:|  :--:  |:--:              |:--:    | :--:   |
| DP | 76.776 | 65.850           | 75.954 | 67.642 |
|DDP | 53.176 | 43.683           | 49.138 | 38.339 |


- `DP`: `torch.nn.DataParallel`
- `AMP`: `torch.cuda.amp`
- `DDP`: `torch.nn.parallel.DistributedDataParallel`
- `cudnn_benchmark`: `torch.backends.cudnn.benchmark = True`

## ENV
```json
{
    "Python":"3.9.5",
    "torch":"1.8.1",
    "torchvision":"0.9.1",
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

## Usage
```bash
# vanilla
python baseline.py --workers 8
# DP + AMP
python baseline.py --workers 8 --dp --amp
# DP + cudnn_benchmark
python baseline.py --workers 8 --dp  --cudnn_benchmark
# DDP
python -m torch.distributed.launch --nproc_per_node=2 faster.py --workers 8
# DDP + AMP
python -m torch.distributed.launch --nproc_per_node=2 faster.py --workers 8 --amp
# DDP + cudnn_benchmark
python -m torch.distributed.launch --nproc_per_node=2 faster.py --workers 8 --cudnn_benchmark
```