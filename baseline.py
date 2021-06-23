from utils.utils import Benchmark
from models import *
import argparse
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from utils import make_logger
from torch.utils.data import DataLoader

mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

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

        train_loss += loss.item()

        if ((i + 1) % 90 == 0):
            logger.info(f'Train Epoch # {epoch} [{i:>5}/{len(train_loader)}] \tlr: {optimizer.param_groups[0]["lr"]} \tloss: {train_loss / 90:>7.6f}')
            train_loss = 0


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
    parser.add_argument('--dp',                default=False, action=argparse.BooleanOptionalAction)
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
    
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10(
            train=True,
            download=False,
            root='./data',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    logger.info(f'Transform:\n{train_loader.dataset.transform}')

    test_loader = DataLoader(
        torchvision.datasets.CIFAR10(
            train=False,
            download=False,
            root='./data',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    net = ThinNet(in_channels=3, filters=[32, 64, 128], n_blocks=[2, 2, 2], n_layers=[1, 1, 1], bn=True)
    net.to(device)
    if args.dp:
        net = torch.nn.DataParallel(net)
        logger.info(f'use gpus: {net.device_ids}')

    logger.info(f'Model: \n{net}')
    params_num = sum(p.numel() for p in net.parameters())
    logger.info(f'Params: {params_num} ({(params_num * 4) / (1024 * 1024):>7.4f}MB)')
        
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    benchmark = Benchmark(logger=logger)
    for epoch in range(args.epochs):
        train(train_loader, net, criterion, optimizer, epoch, args)
        test(test_loader, net, epoch, args)
    benchmark.elapsed()

