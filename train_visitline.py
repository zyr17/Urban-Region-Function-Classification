import torch
import torchvision
import time
import numpy as np
import pickle
import random

torch.manual_seed(19951017)
random.seed(19951017)

label_num = 9

def cuda(tensor):
    """
    A cuda wrapper
    """
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

class CNN(torch.nn.Module):
    def __init__(self, inputlen = (7, 26, 24), outputlen = label_num):
        super(CNN, self).__init__()
        self.visit_convs = torch.nn.ModuleList()
        self.visit_fcs = torch.nn.ModuleList()
        self.visit_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(inputlen[0], 100, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ))# [B, 100, 13, 12]
        self.visit_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(100, 200, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        ))# [B, 200, 7, 6]
        self.visit_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(200, 400, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        ))# [B, 400, 4, 3]
        self.visit_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(400, 800, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(0, 1))
        ))# [B, 800, 2, 2]
        self.visit_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(800 * 2 * 2, 160),
            torch.nn.ReLU()
        ))# [B, 160]
        self.visit_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(160, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images):
        x = inputs # [B, 7, 26, 24]
        for conv in self.visit_convs:
            x = conv(x)
        x = x.reshape(-1, 800 * 2 * 2)
        for fc in self.visit_fcs:
            x = fc(x)
        return x

class LineDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y).float()
    def __getitem__(self, index):
        xarr = np.zeros(26 * 7 * 24)
        for i in self.x[index]:
            xarr[i] = 1
        return xarr, self.y[index]
    def __len__(self):
        return len(self.x1)

    def collate_fn(self, data):
        x, y = list(zip(*data))
        x = cuda(torch.tensor(np.concatenate(x)).float())
        y = cuda(torch.tensor(y).float())
        return x, y

[train_x, train_y] = pickle.load(open('data/pickle/visitline_23.pkl', 'rb'))

print('read done')

zipped = list(zip(train_x, train_y))
print(len(zipped), zipped[0])
random.shuffle(zipped)
print('random')
[train_x, train_y] = list(zip(*zipped))
print('zip2')
train_y = np.array(train_y, dtype='int8')
print(len(train_x), len(train_y))
print(train_x[0], train_y[0])

pickle.dump([train_x, train_y], open('data/pickle/visitline_23_shuffle.pkl', 'wb'))