import argparse
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from utils import *
from models import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    train_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.shape[0]

    logger.info(f'Train Epoch # {epoch} [{i:>5}/{len(train_loader)}] \tloss: {train_loss / len(train_loader.dataset):>7.6f}')


def test(test_loader, model, epoch, args):
    model.eval()

    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            output = model(images)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum()

        logger.info(f'\tTest Epoch #{epoch:>2}: {correct}/{len(test_loader.dataset)} ({100. * correct.item() / len(test_loader.dataset):>3.2f}%)')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cudnn_benchmark',    default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dp',                 default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--amp',                default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-j', '--workers',      type=int,   default=8)
    parser.add_argument('--epochs',             type=int,   default=10)
    parser.add_argument('-b', '--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.001)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--output-dir',         type=str,   default='logs')
    return parser.parse_args()

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')
  
    args = parse_args()
    logger = make_logger('cifar_10', 'logs')

    logger.info(args)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    train_loader = DataLoader(
        CIFAR10('./data', True,  T.ToTensor(), download=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        CIFAR10('./data', False, T.ToTensor()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    net = ThinNet(in_channels=3, filters=[32, 64, 128], n_blocks=[2, 2, 2], n_layers=[1, 1, 1])
    net.to(device)
    if args.dp:
        net = torch.nn.DataParallel(net)
        logger.info(f'use gpus: {net.device_ids}')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    benchmark = Benchmark(logger=logger)
    for epoch in range(args.epochs):
        train(train_loader, net, criterion, optimizer, epoch, args)
        test(test_loader, net, epoch, args)
    benchmark.elapsed()

