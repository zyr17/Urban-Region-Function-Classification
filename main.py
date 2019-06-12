import pickle
import torch
import torchvision
import numpy as np
import time

torch.manual_seed(19951017)
label_num = 9

with open('data/pickle/train_visit.bucket.pkl', 'rb') as f:
    train_visit = pickle.load(f)
with open('data/pickle/train_label.pkl', 'rb') as f:
    train_label_raw = pickle.load(f)
train_label = np.zeros((len(train_label_raw), label_num), dtype='float')
for num in range(len(train_label_raw)):
    train_label[num][train_label_raw[num] - 1] = 1
with open('data/pickle/train_image.pkl', 'rb') as f:
    train_image = pickle.load(f) # TODO: use image

with open('data/pickle/test_visit.bucket.pkl', 'rb') as f:
    test_visit = pickle.load(f)
with open('data/pickle/test_image.pkl', 'rb') as f:
    test_image = pickle.load(f)

def change_visit_shape(visit):
    visit = np.array(visit, dtype='int32')
    visit = visit.reshape(-1, 26, 7, 24)
    visit = visit.transpose(0, 2, 1, 3)
    visit = visit.reshape(-1, 26 * 7 * 24)
    return visit

train_visit = change_visit_shape(train_visit)
test_visit = change_visit_shape(test_visit)
train_image = train_image.transpose(0, 3, 1, 2)
test_image = test_image.transpose(0, 3, 1, 2)
'''
train_visit = train_visit[:200]
train_label = train_label[:200]
train_image = train_image[:200]
test_visit = test_visit[:100]
test_image = test_image[:100]
'''
# use last 10% data as validation set
val_line = len(train_visit) // 10 * 9
val_visit = train_visit[val_line:]
val_label = train_label[val_line:]
val_image = train_image[val_line:]
train_visit = train_visit[:val_line]
train_label = train_label[:val_line]
train_image = train_image[:val_line]

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

class FC(torch.nn.Module):
    def __init__(self, inputlen = 26 * 7 * 24, outputlen = label_num, hidden = [1000, 200, 50]):
        super(FC, self).__init__()
        self.nnum = [inputlen]
        for i in hidden:
            self.nnum.append(i)
        self.nnum.append(outputlen)
        self.linear = torch.nn.ModuleList()
        for i in range(len(self.nnum) - 1):
            self.linear.append(torch.nn.Linear(self.nnum[i], self.nnum[i + 1]))
        self.Sigmoid = torch.nn.Sigmoid()
        self.ReLU = torch.nn.ReLU()
    def forward(self, inputs, images):
        x = inputs # [B, inputlen]
        for num, linear in enumerate(self.linear):
            if num == len(self.linear) - 1:
                x = self.Sigmoid(linear(x))
            else:
                x = self.ReLU(linear(x))
        return x # [B, outputlen]

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
#TODO: MaxPool2d在不整除时行为
class concat_after(torch.nn.Module):
    def __init__(self, visitlen = (7, 26, 24), imagelen = (3, 100, 100), outputlen = label_num):
        super(concat_after, self).__init__()
        self.visit_convs = torch.nn.ModuleList()
        self.visit_fcs = torch.nn.ModuleList()
        self.visit_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(visitlen[0], 100, 5, padding=2),
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

        self.image_convs = torch.nn.ModuleList()
        self.image_fcs = torch.nn.ModuleList()
        self.image_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(imagelen[0], 50, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ))# [B, 50, 50, 50]
        self.image_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(50, 200, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        ))# [B, 200, 25, 25]
        self.image_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(200, 400, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, padding=1)
        ))# [B, 400, 9, 9]
        self.image_convs.append(torch.nn.Sequential(
            torch.nn.Conv2d(400, 800, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3)
        ))# [B, 800, 3, 3]
        self.image_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(800 * 3 * 3, 240),
            torch.nn.ReLU()
        ))# [B, 240]

        self.final_fcs = torch.nn.ModuleList()
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(160 + 240, 80),
            torch.nn.ReLU()
        ))# [B, 80]
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(80, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images):
        x = inputs # [B, 7, 26, 24]
        for conv in self.visit_convs:
            x = conv(x)
        x = x.reshape(-1, 800 * 2 * 2)
        for fc in self.visit_fcs:
            x = fc(x)
        
        y = images
        for conv in self.image_convs:
            y = conv(y)
        y = y.reshape(-1, 800 * 3 * 3)
        for fc in self.image_fcs:
            y = fc(y)
        
        z = torch.cat([x, y], 1)
        for fc in self.final_fcs:
            z = fc(z)
        
        return z

class VisitDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = torch.tensor(x1).float()
        self.x2 = torch.tensor(x2).float()
        self.y = torch.tensor(y).float()
    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]
    def __len__(self):
        return len(self.x1)

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

batch_size = 10
epoch_number = 100
learning_rate = 0.00001
#modelname = 'CNN'
modelname = 'concat_after'

if (modelname != 'FC'):
    train_visit = train_visit.reshape(-1, 7, 26, 24)
    val_visit = val_visit.reshape(-1, 7, 26, 24)
    test_visit = test_visit.reshape(-1, 7, 26, 24)
train_dataset = VisitDataset(train_visit, train_image, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
test_dataset = VisitDataset(test_visit, test_image, np.zeros(len(test_visit)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, False)
val_dataset = VisitDataset(val_visit, val_image, val_label)
val_loader = torch.utils.data.DataLoader(val_dataset, 100, True)
if (modelname == 'CNN'):
    model = CNN()
elif (modelname == 'FC'):
    model = FC()
elif modelname == 'concat_after':
    model = concat_after()
model = cuda(model)
loss = torch.nn.MSELoss()

print('start training')
for epoch in range(epoch_number):
    print('epoch', epoch)
    start_time = time.clock()
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    model.train()
    batch_count = 0
    for input, image, label in train_loader:
        input = cuda(input)
        image = cuda(image)
        label = cuda(label)
        opt.zero_grad()
        pred = model(input, image)
        L = loss(pred, label)
        if (batch_count % (len(train_label) // 10 // batch_size) == 0):
            print(batch_count, L.data.item())
        L.backward()
        opt.step()
        batch_count += 1
    model.eval()
    correct = 0
    for input, image, label in val_loader:
        input = cuda(input)
        image = cuda(image)
        label = cuda(label)
        pred = model(input, image)
        for i in range(len(input)):
            if Accuracy(pred[i], label[i]):
                correct += 1
    correct /= len(val_visit)
    print(correct, 'use time:', time.clock() - start_time)

    result = []
    num = 0
    for input, image, xxx in test_loader:
        preds = model(cuda(torch.tensor(input).float()), cuda(torch.tensor(image).float()))
        #print(num, input)
        oneres = 0
        #print(num, pred)
        for pred in preds:
            for i in range(len(pred)):
                if pred[i] > pred[oneres]:
                    oneres = i
            result.append(str(num).zfill(6) + '\t00' + str(oneres + 1))
            num += 1
    with open('data/results/' + modelname + '_' + str(epoch).zfill(5) + '_' + str(correct) +  '.txt', 'w') as f:
        f.write('\n'.join(result))
