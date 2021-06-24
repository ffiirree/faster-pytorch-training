#!/bin/bash -x
python baseline.py --epochs $1
python baseline.py --epochs $1

python baseline.py --epochs $1 --dp
python baseline.py --epochs $1 --dp
python baseline.py --epochs $1 --dp --cudnn_benchmark
python baseline.py --epochs $1 --dp --amp
python baseline.py --epochs $1 --dp --cudnn_benchmark --amp

python -m torch.distributed.launch --nproc_per_node=2 faster.py --epochs $1
python -m torch.distributed.launch --nproc_per_node=2 faster.py --epochs $1 --amp
python -m torch.distributed.launch --nproc_per_node=2 faster.py --epochs $1 --cudnn_benchmark
python -m torch.distributed.launch --nproc_per_node=2 faster.py --epochs $1 --cudnn_benchmark --amp