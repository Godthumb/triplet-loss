#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from loss import TripletLoss
import torch.nn.functional as F
import argparse
from dataset import TripletDataSet

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/data/path')
    parser.add_argument('--arch', type=str, default='/data/path')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--data-loader-workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    return args

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

def main(opt):
    best_loss = 1000

    # create model
    print("=> creating model '{}'".format(opt.arch))
    if opt.arch.lower().startswith('resnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model = model.resnet18(pretrained=True, num_classes=1000)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Sequential(nn.Linear(512, 128, bias=False), L2_norm())
    else:
        raise ValueError('Wrong arch')

    # a customized resnet model with last feature map size as 14x14 for better class activation mapping

    model = model.cuda()
    print (model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            start_epoch = opt.start_epoch
    # Data loading code
    traindir = os.path.join(opt.data_path, 'train')
    valdir = os.path.join(opt.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr_Transform = transforms.Compose([
                                      # transforms.Lambda(lambda img:_cloud_crop(img)),
                                      # transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
                                      transforms.RandomResizedCrop(224),
                                      # transforms.CenterCrop(336),
                                      transforms.RandomHorizontalFlip(),
                                      # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      normalize,])
    train_dataset = TripletDataSet(traindir, tr_Transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=opt.batch_size, 
                                               shuffle=True,
                                               num_workers=6)

    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,])
    val_dataset = TripletDataSet(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=opt.batch_size, 
                                             shuffle=False,
                                             num_workers=6)   

    # define loss function (criterion) and pptimizer
    criterion_triple = TripletLoss(device='cuda: 0')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if opt.evaluate:
        validate(val_loader, model, criterion_triple)
        return

    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion_triple, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion_triple)

        # remember best prec@1 and save checkpoint
        is_best = val_loss < best_loss
        best_loss = max(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, epoch, opt.arch.lower())


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    model.train()
    train_loader_length = len(train_loader)
    end = time.time()
    for i, sample in enumerate(train_loader):
        input, target = sample
        input_var, target_var = input.cuda(), target.cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        output = model(input_var)  # output is feature
      
        loss = criterion(target_var, output)
     
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

   
    print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
        epoch, i, train_loader_length, batch_time=batch_time,
        data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var, target_var = input.cuda(), target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), input_var.size(0))
  
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
    print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        .format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        ))

    return losses.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth'):
    if epoch != 0:
        os.rename(filename + '_latest.pth', filename + '_%d.pth' % (epoch))
    torch.save(state, filename + '_latest.pth')
    if is_best:
        shutil.copyfile(filename + '_latest.pth', filename + '_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr_ = lr * (0.5 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

if __name__ == '__main__':
    args = opt()
    main(args)