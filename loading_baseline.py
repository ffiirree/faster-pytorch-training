import argparse
import os

import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms

from utils import *

logger = make_logger('imagenet', 'logs')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--local_rank', metavar='RANK', type=int, default=0)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    return parser.parse_args()


def train(train_loader, epoch, args):
    load_time = Benchmark()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)

        logger.info(f'Epoch #{epoch} [{i}/{len(train_loader)}] {load_time.elapsed():>.3f}')

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'

    args = parse_args()
    args.batch_size = int(args.batch_size / torch.cuda.device_count())

    logger.info(args)

    dist.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    # Data loading code
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    benchmark = Benchmark()
    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, epoch, args)
    logger.info(f'{benchmark.elapsed():>.3f}')
