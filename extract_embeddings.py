import torch
from dataset import TripletDataSet
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import model
import os
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


if __name__ == '__main__':
    # choose cuda or cpu device
    device = 0
    torch.cuda.set_device(device)

    # dataloader
    batch_size = 64
    data_path = r'D:\BaiduNetdiskDownload\catsdogs'

    # load model
    model = model.resnet18(pretrained=False, num_classes=1000)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(nn.Linear(512, 128, bias=False), L2_norm())
    checkpoint = torch.load('resnet18_triplet_33_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    # load data
    traindir = os.path.join(data_path, 'train')                                                                                                                                                                                                                                                                                                                                                                                             
    valdir = os.path.join(data_path, 'test')
    x_val = []
    y_val = []
    x_train = []
    y_train = []
    # n_class = 0
 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,])
    val_dataset = TripletDataSet(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             num_workers=0)
    
    # use same transform as val
    tr_dataset = TripletDataSet(traindir, val_transform)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             num_workers=0)
    for (input, target) in val_loader:
        input_var, target_var = input.cuda(), target.cuda()
        with torch.no_grad():
            embedding = model(input_var)
        x_val.append(embedding.cpu().numpy())
        y_val.append(target_var.cpu().numpy())

    for (input, target) in tr_loader:
        input_var, target_var = input.cuda(), target.cuda()
        with torch.no_grad():
            embedding = model(input_var)
        x_train.append(embedding.cpu().numpy())
        y_train.append(target_var.cpu().numpy())


    x_val, y_val = np.concatenate(x_val, 0), np.concatenate(y_val, 0)
    x_train, y_train = np.concatenate(x_train, 0), np.concatenate(y_train, 0)

    np.save('x_train', x_train)
    np.save('y_train', y_train)
    np.save('x_val', x_val)
    np.save('y_val', y_val)

    x_tsne = TSNE(n_components=2).fit_transform(x_val)
    print(x_tsne.shape)

    color = ['darkcyan', 'r']
    labels = [str(i) for i in range(2)]
    times = [0 for i in range(2)]
    for i in range(len(x_tsne)):
        if times[y_val[i]] == 0:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_val[i]], label=labels[y_val[i]])
            times[y_val[i]] += 1
        else:
            plt.scatter(x_tsne[i, 0], x_tsne[i, 1], color=color[y_val[i]])
    plt.legend()
    plt.title('TSNE validation')
    plt.savefig('tsne_val.png')