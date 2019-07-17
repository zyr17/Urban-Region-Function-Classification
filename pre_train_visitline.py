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
        self.y = np.array(y, dtype='float')
    def __getitem__(self, index):
        xarr = np.zeros(26 * 7 * 24)
        for i in self.x[index]:
            xarr[i] = 1
        xarr = xarr.reshape(26, 7, 24)
        xarr = xarr.transpose(1, 0, 2)
        return xarr, self.y[index]
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

if __name__ == '__main__':
    #filename = 'data/pickle/part/visitline_23/test.pkl'
    filename = 'data/pickle/train_visitline_23_shuffle.pkl'

    train_line, raw_label = pickle.load(open(filename, 'rb'))
    train_label = np.zeros((len(raw_label), label_num))
    for num, i in enumerate(raw_label):
        train_label[num][i - 1] = 1
    print('pickle load over')

    #train_line = train_line[:1000]
    #train_label = train_label[:1000]
    #pickle.dump([train_line, raw_label[:1000]], open('data/pickle/part/visitline_23/test.pkl', 'wb'))

    val_num = len(train_line) // 200 * 199
    val_line = train_line[val_num:]
    val_label = train_label[val_num:]
    train_line = train_line[:val_num]
    train_label = train_label[:val_num]

    batch_size = 100
    epoch_number = 100
    learning_rate = 0.0001

    train_dataset = LineDataset(train_line, train_label)
    val_dataset = LineDataset(val_line, val_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, True, collate_fn=val_dataset.collate_fn)
    model = cuda(CNN())
    loss = torch.nn.MSELoss()

    print('start training')
    last_save_time = time.time()
    last_batch_time = time.time()
    save_interval = 900
    batch_interval = 60
    save_count = 0

    #save_count = 42
    #model.load_state_dict(torch.load('data/models/visitline/%04d_8_74089_0.4788.pkl' % (save_count,)))

    for epoch in range(epoch_number):
        print('epoch', epoch)
        opt = torch.optim.Adam(model.parameters(), learning_rate)
        batch_count = 0
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
                for input, label in val_loader:
                    pred = model(input)
                    for i in range(len(input)):
                        if Accuracy(pred[i], label[i]):
                            correct += 1
                correct /= len(val_line)
                print(time.time() - last_save_time, correct)
                torch.save(model.state_dict(), 'data/models/visitline/%04d_%d_%d_%.4f.pkl' % (save_count, epoch, batch_count, correct))
                last_save_time = time.time()
