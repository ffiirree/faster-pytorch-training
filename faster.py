from utils.utils import Benchmark
from models import *
import argparse
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.distributed as dist
from utils import make_logger
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    train_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.shape[0]

    logger.info(f'Train Epoch # {epoch}@{args.local_rank} [{i:>5}/{len(train_loader)}] \tloss: {train_loss / len(train_loader.dataset):>7.6f}')


def test(test_loader, model, epoch, args):
    model.eval()

    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            output = model(images)

            _, predicted = torch.max(output.data, 1)

            correct += (predicted == labels).sum()

        dist.all_reduce(correct)
        if args.local_rank == 0:
            logger.info(f'\tTest Epoch #{epoch:>2}: {correct}/{len(test_loader.dataset)} ({100. * correct.item() / len(test_loader.dataset):>3.2f}%)')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cudnn_benchmark',    default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--amp',                default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--local_rank',         type=int, default=0)
    parser.add_argument('-j', '--workers',      type=int,   default=None)
    parser.add_argument('--epochs',             type=int,   default=10)
    parser.add_argument('-b', '--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.001)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--output-dir',         type=str,   default='logs')
    return parser.parse_args()

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
  
    args = parse_args()
    logger = make_logger('cifar_10', 'logs')

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group('nccl')
    device = torch.device(f'cuda:{args.local_rank}')

    logger.info(args)
    
    if args.local_rank == 0:
        CIFAR10('./data', True, download=True)
    dist.barrier()

    train_dataset = CIFAR10('./data', True, T.ToTensor())
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    test_dataset = CIFAR10('./data', False, T.ToTensor())
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler
    )

    net = ThinNet(in_channels=3, filters=[32, 64, 128], n_blocks=[2, 2, 2], n_layers=[1, 1, 1])
    # net.load_state_dict(torch.load(f'{args.output_dir}/cifar10.pt'))
    net.to(device)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    benchmark = Benchmark(args.local_rank == 0, logger=logger)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, net, criterion, optimizer, epoch, args)
        test(test_loader, net, epoch, args)
    benchmark.elapsed()

    # if args.local_rank == 0:
    #     model_name = f'{args.output_dir}/cifar10.pt'
    #     torch.save(net.module.state_dict(), model_name)
    #     logger.info(f'Saved: {model_name}!')
