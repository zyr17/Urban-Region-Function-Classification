import torch
import torchvision
import time
import numpy as np
import pickle
import random
import os
import json

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
        self.y = y
    def __getitem__(self, index):
        xarr = np.zeros(26 * 7 * 24)
        for i in self.x[index]:
            xarr[i] = 1
        xarr = xarr.reshape(26, 7, 24)
        xarr = xarr.transpose(1, 0, 2)

        is_MSE = True
        if is_MSE:
            yres = np.zeros(label_num, dtype='float')
            yres[self.y[index] - 1] = 1
        else:
            yres = self.y[index]

        return xarr, yres
    def __len__(self):
        return len(self.x)

    def collate_fn(self, data):
        x, y = list(zip(*data))
        #print(len(x), len(y), x[0], y[0], type(x[0]), type(y[0]))
        x = cuda(torch.tensor(x).float())
        y = cuda(torch.tensor(y).float())
        return x, y

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

def read_val_data(filenames, savename, val_part = 0.001):
    if os.path.exists(savename):
        x, y = pickle.load(open(savename, 'rb'))
    else:
        x = []
        y = []
        for filename in filenames:
            print('read_val_data', filename)
            X, Y = pickle.load(open(filename, 'rb'))
            val_line = int(len(X) * (1 - val_part))
            x += X[val_line:]
            y.append(Y[val_line:])
        y = np.concatenate(y)
        print(type(x), type(y), len(x), len(y), y.shape, y.dtype)
        pickle.dump([x, y], open(savename, 'wb'))
    dataset = LineDataset(x, y)
    return torch.utils.data.DataLoader(dataset, 100, False, collate_fn=dataset.collate_fn)

def read_data(filename, batch_size = 100, val_part = 0.001):
    print('read_data', filename)
    x, y = pickle.load(open(filename, 'rb'))
    val_line = int(len(x) * (1 - val_part))
    x = x[:val_line]
    y = y[:val_line]
    dataset = LineDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size, True, collate_fn=dataset.collate_fn)

if __name__ == '__main__':
    #filename = 'data/pickle/part/visitline_23/test.pkl'
    #filename = 'data/pickle/train_visitline_23_shuffle.pkl'
    filenames = ['data/pickle/visitline_train_shuffle/c_%06d_%06d.pkl' % (x, x + 100000) for x in range(0, 400000, 100000)]

    val_loader = read_val_data(filenames, 'data/pickle/visitline_train_shuffle/val_000000_400000.pkl')

    batch_size = 1024
    epoch_number = 100
    learning_rate = 0.0001

    #train_dataset = LineDataset(train_line, train_label)
    #val_dataset = LineDataset(val_line, val_label)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, collate_fn=train_dataset.collate_fn)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, True, collate_fn=val_dataset.collate_fn)
    model = cuda(CNN())
    loss = torch.nn.MSELoss() # TODO: CrossEntropyLoss

    print('start training')
    last_save_time = time.time()
    last_batch_time = time.time()
    save_interval = 900
    batch_interval = 60
    save_count = 0
    last_epoch = 0
    save_folder = 'data/models/visitline_' + time.strftime('%d_%H%M%S', time.localtime()) + '/'
    #save_folder = 'data/models/visitline/'

    #load trained models
    save_count = 107
    last_epoch = 0
    filenames = filenames[last_epoch % len(filenames):] + filenames[:last_epoch % len(filenames)]
    save_folder = 'data/models/visitline_15_161309/'
    model.load_state_dict(torch.load('%s%04d.model' % (save_folder, save_count)))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        open(save_folder + 'detail.txt', 'a').write('  id epoc    batch  correct\n')

    for epoch in range(last_epoch, epoch_number):
        print('epoch', epoch)
        opt = torch.optim.Adam(model.parameters(), learning_rate)
        batch_count = 0

        train_loader = read_data(filenames[0], batch_size)
        filenames = filenames[1:] + filenames[:1]

        for input, label in train_loader:
            model.train()
            opt.zero_grad()
            pred = model(input)
            L = loss(pred, label)
            if (time.time() - last_batch_time > batch_interval):
                print(batch_count, L.data.item())
                last_batch_time = time.time()
            L.backward()
            opt.step()
            batch_count += 1
            if time.time() - last_save_time > save_interval:
                # after some time, eval and save model
                save_count += 1
                print('start eval #%04d' % (save_count))
                model.eval()
                correct = 0
                val_number = 0
                for input, label in val_loader:
                    pred = model(input)
                    val_number += len(input)
                    for i in range(len(input)):
                        if Accuracy(pred[i], label[i]):
                            correct += 1
                correct /= val_number
                print(time.time() - last_save_time, correct)
                torch.save(model.state_dict(), save_folder + '%04d.model' % (save_count,))
                open(save_folder + 'detail.txt', 'a').write('%04d %04d %08d %.6f\n' % (save_count, epoch, batch_count, correct))
                last_save_time = time.time()
