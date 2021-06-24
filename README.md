# Faster

Single-machine multi-GPU 

## Run Time
```
Batch size: 512, conv layers: 11, epochs: 5
```
> Baseline : 276.980s

|    |         | +cudnn_benchmark | +AMP   | +cudnn_benchmark +AMP |
|:--:|  :--:   |:--:              |:--:    | :--:                  |
| DP | 163.740 | 104.807          | 74.948 | 73.862                |
|DDP | 142.497 | 102.535          | 67.095 | 72.998                |



- `DP`: `torch.nn.DataParallel`
- `AMP`: `torch.cuda.amp`
- `DDP`: `torch.nn.parallel.DistributedDataParallel`
- `cudnn_benchmark`: `torch.backends.cudnn.benchmark = True`
- `pin_memory=True`
- `non_blocking=True`
- `optimizer.zero_grad(set_to_none=True)`

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
# $1 is the epochs
./run_all.sh 5
```