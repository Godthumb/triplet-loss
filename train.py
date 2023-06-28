#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from loss import TripletLoss
import model
import torch.nn.functional as F
# import inceptionv3
# import BilinearCNN
# from zrmodel import LeNet32

from dataset import TripletDataSet

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# 存放数据的根目录
DATA_PATH = '/home/meteo/zihao.chen/data'

# 数据集的目录，它内部应该满足pytorch Dataset的标准，形如 data/class1/*.jpg data/class2/*.jpg data/class3/*.jpg
data_path = os.path.join(DATA_PATH, 'allergy/allergy_work')

# 最优迭代的索引，初始为0
best_loss = 1000

# 一个特立独行的名字，用来区分每次训练的模型
arch = 'resnet18_triplet_base'

# 是否使用现有的模型,如果使用的话，注意你的dataloader是否能够正确的投喂数据
# resume = 'resnet50_cloud_a29_best.pth.tar'
resume = None

# 模型的分类类别数量
num_classes = 10

# 每批次训练的样本量，也会影响dataloader的缓冲大小
batch_size = 10

# dataloader 使用的线程数量，之所以没用进程的方式，是因为主要是大多数数据装载的时间主要集中在IO阻塞上，
# 数据的预处理本身占用的时间其实很快。而且进程间通讯和调度没有线程那么方便。
data_loader_workers = 8

# 是否是用来执行评价过程的,看代码，很简单
evaluate = False

# 调参的参数
lr = 0.0001
momentum = 0.9
weight_decay = 1e-4

# 多少个batch打印一次结果
print_freq = 10

# 迭代开始的索引
start_epoch = 0

# 总数据集迭代次数
epochs = 40


def _cloud_crop(img):
    """
    :param img:输入图像
    :return: 返回截取后的图像
    """
    w, h = img.size
    if w > 100 and h > 100:
        lw = int(w * 0.1)
        lh = int(h * 0.2)
        return img.crop((lw, 0, w - lw, h - lh))
    return img

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

def main():
    global best_prec1
    global start_epoch

    # create model
    print("=> creating model '{}'".format(arch))
    if arch.lower().startswith('resnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model = model.resnet18(pretrained=True, num_classes=1000)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Sequential(nn.Linear(512, 128, bias=False), L2_norm())
       
    elif arch.lower().startswith('bcnn'):
        model = BilinearCNN.BilinearCNN(num_classes)
    elif arch.lower().startswith('inception'):
        # model = inceptionv3.inception_v3(pretrained=True)
        model = inceptionv3.inception_v3(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif arch.lower().startswith('zrmodel'):
        model = LeNet32(1)
    else:
        model = models.__dict__[arch](num_classes=365)
        state_dict = torch.load('whole_alexnet_places365_python36.pth')
        model.load_state_dict(state_dict)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # a customized resnet model with last feature map size as 14x14 for better class activation mapping

    model = model.cuda()
    print (model)

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
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
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers=6)

    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,])
    val_dataset = TripletDataSet(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             num_workers=6)   

    # define loss function (criterion) and pptimizer
    criterion_triple = TripletLoss(device='cuda: 0')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if evaluate:
        validate(val_loader, model, criterion_triple)
        return

    for epoch in range(start_epoch, epochs):
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
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, epoch, arch.lower())


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
     
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), anchor.size(0))
        # top5.update(prec5.item(), anchor.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                epoch, i, train_loader_length // batch_size, batch_time=batch_time,
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

        if i % print_freq == 0:
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


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     # print (pred)
#     # print (target)
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


if __name__ == '__main__':
    main()