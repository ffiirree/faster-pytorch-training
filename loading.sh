#!/bin/bash -x
python -m torch.distributed.launch --nproc_per_node=2 $1 --workers 8 --dali_cpu --epochs 1 $2
