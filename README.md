# Faster

Single-machine multi-GPU 

## Run Time
|| vanilla | DP | DP+AMP | DDP | DDP+AMP |
|:--:|:--:|:--:|:--:| :--: |:--:|
| 10 epcohs/s| 82.95 | 78.88 | 77.45 | 44.00 | 37.96 |

- `vanilla`: 1 GPU
- `DP`: 2 GPUs with `torch.nn.DataParallel`
- `DP + AMP`: 2 GPUs with `torch.nn.DataParallel` and `torch.cuda.amp`
- `DDP`: 2 GPUs with `torch.nn.parallel.DistributedDataParallel`
- `DDP + AMP`: 2 GPUs with `torch.nn.parallel.DistributedDataParallel` and `torch.cuda.amp`

## Usage
```bash
# vanilla
python baseline.py --workers 8
# DP
python baseline.py --workers 8 --dp
# DP + AMP
python baseline.py --workers 8 --dp  --amp
# DDP
python -m torch.distributed.launch --nproc_per_node=2 faster.py --workers 8
# DDP + AMP
python -m torch.distributed.launch --nproc_per_node=2 faster.py --workers 8 --amp
```