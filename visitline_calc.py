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
    def forward(self, inputs):
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
        self.y = np.array(y, dtype='int32')
    def __getitem__(self, index):
        xarr = np.zeros(26 * 7 * 24)
        for i in self.x[index]:
            xarr[i] = 1
        xarr = xarr.reshape(26, 7, 24)
        xarr = xarr.transpose(1, 0, 2)
        return xarr, self.y[index], len(self.x[index])
    def __len__(self):
        return len(self.x)

    def collate_fn(self, data):
        x, y, z = list(zip(*data))
        #print(len(x), len(y), x[0], y[0], type(x[0]), type(y[0]))
        x = cuda(torch.tensor(x).float())
        #y = torch.tensor(y)
        return x, y, z

def Accuracy(x, y):
    xi = 0
    yi = 0
    for i in range(len(x)):
        if x[xi] < x[i]:
            xi = i
    for i in range(len(y)):
        if y[yi] < y[i]:
            yi = i
    return xi == yi

def read_data(filename, batch_size = 100, val_part = 0.001):
    print('read_data', filename)
    x, y = pickle.load(open(filename, 'rb'))
    val_line = int(len(x) * (1 - val_part))
    #print(len(x), val_line)
    x = x[:val_line]
    y = y[:val_line]
    dataset = LineDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size, True, collate_fn=dataset.collate_fn)

if __name__ == '__main__':
    filenames = ['data/pickle/visitline_test/c_000000_100000.pkl']
    #filenames = ['data/pickle/visitline_train_1/c_%06d_%06d.pkl' % (x, x + 100000) for x in range(0, 400000, 100000)]

    savenames = ['data/pickle/visitline_test_res/c_000000_100000.pkl']
    #savenames = ['data/pickle/visitline_train_res/c_%06d_%06d.pkl' % (x, x + 100000) for x in range(0, 400000, 100000)]

    modelname = 'data/models/visitline_15_161309/0012.pkl'

    #train_line, train_id = pickle.load(open(filename, 'rb'))
    #print('pickle load over')

    #train_line = train_line[9000:10000]
    #train_id = train_id[9000:10000]
    #pickle.dump([train_line, raw_label[:1000]], open('data/pickle/part/visitline_23/test.pkl', 'wb'))

    batch_size = 1024

    for index, filename, savename in zip(range(0, 100000 * len(filenames), 100000), filenames, savenames):
        print(filename, savename)

        train_loader = read_data(filename, batch_size, 0)
        model = cuda(CNN())

        print('start eval')

        model.load_state_dict(torch.load(modelname))

        res = [[] for x in range(100000)]

        count = 0

        for input, label, lengths in train_loader:
            count += 1
            if count % 10 == 0:
                print(count)
            model.eval()
            pred = model(input)
            for num in range(len(pred)):
                l = label[num] - index
                length = lengths[num]
                p = pred[num]
                #print(l, length, p)
                res[l].append([length] + p.tolist())

        for num in range(len(res)):
            res[num] = np.array(res[num], dtype='float')

        pickle.dump(res, open(savename, 'wb'))
