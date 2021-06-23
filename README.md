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