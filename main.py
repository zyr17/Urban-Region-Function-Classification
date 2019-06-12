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

#train_visit = train_visit[:10000]
#train_label = train_label[:10000]
#train_image = train_image[:10000]

# use last 10% data as validation set
val_line = len(train_visit) // 10 * 9
val_visit = train_visit[val_line:]
val_label = train_label[val_line:]
val_image = train_image[val_line:]
train_visit = train_visit[:val_line]
train_label = train_label[:val_line]
train_image = train_image[:val_line]

def change_visit_shape(visit):
    visit = np.array(visit, dtype='int32')
    visit = visit.reshape(-1, 26, 7, 24)
    visit = visit.transpose(0, 2, 1, 3)
    visit = visit.reshape(-1, 26 * 7 * 24)
    return visit

train_visit = change_visit_shape(train_visit)
val_visit = change_visit_shape(val_visit)
test_visit = change_visit_shape(test_visit)

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
    def forward(self, inputs):
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
        self.conv1 = torch.nn.Conv2d(inputlen[0], 100, 5, padding=2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(inputlen[0], 100, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(100, 200, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(200, 400, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(400, 800, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(0, 1))
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(800 * 2 * 2, 160),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(160, outputlen),
            torch.nn.Sigmoid()
        )
    def forward(self, inputs):
        x = inputs # [B, 7, 26, 24]
        #print(x.size())
        x = self.conv1(x) # [B, 100, 13, 12]
        #print(x.size())
        x = self.conv2(x) # [B, 200, 7, 6]
        #print(x.size())
        x = self.conv3(x) # [B, 400, 4, 3]
        #print(x.size())
        x = self.conv4(x) # [B, 800, 2, 2]
        #print(x.size())
        x = self.fc1(x.reshape(-1, 800 * 2 * 2)) # [B, 160]
        #print(x.size())
        x = self.fc2(x) # [B, 9]
        #print(x.size())
        return x

class concat_after(torch.nn.Module):
    def __init__(self, inputlen = (7, 26, 24), outputlen = label_num):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(inputlen[0], 100, 5, padding=2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(inputlen[0], 100, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(100, 200, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(200, 400, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(1, 0))
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(400, 800, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=(0, 1))
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(800 * 2 * 2, 160),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(160, outputlen),
            torch.nn.Sigmoid()
        )
    def forward(self, inputs):
        x = inputs # [B, 7, 26, 24]
        x = self.conv1(x) # [B, 100, 13, 12]
        x = self.conv2(x) # [B, 200, 7, 6]
        x = self.conv3(x) # [B, 400, 4, 3]
        x = self.conv4(x) # [B, 800, 2, 2]
        x = self.fc1(x.reshape(-1, 800 * 2 * 2)) # [B, 160]
        x = self.fc2(x) # [B, 9]
        return x

class VisitDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

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
modelname = 'CNN'

if (modelname != 'FC'):
    train_visit = train_visit.reshape(-1, 7, 26, 24)
    val_visit = val_visit.reshape(-1, 7, 26, 24)
    test_visit = test_visit.reshape(-1, 7, 26, 24)
train_dataset = VisitDataset(train_visit, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
test_dataset = VisitDataset(test_visit, np.zeros(len(test_visit)))
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size, False)
val_dataset = VisitDataset(val_visit, val_label)
val_loader = torch.utils.data.DataLoader(val_dataset, 100, True)
if (modelname == 'CNN'):
    model = CNN()
elif (modelname == 'FC'):
    model = FC()
elif modelname == 'concat_after':
    model = concat_after()
model = cuda(model)
loss = torch.nn.MSELoss()

for epoch in range(epoch_number):
    print('epoch', epoch)
    start_time = time.clock()
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    model.train()
    batch_count = 0
    for input, label in train_loader:
        input = cuda(input)
        label = cuda(label)
        opt.zero_grad()
        pred = model(input)
        L = loss(pred, label)
        if (batch_count % (len(train_label) // 10 // batch_size) == 0):
            print(batch_count, L.data.item())
        L.backward()
        opt.step()
        batch_count += 1
    model.eval()
    correct = 0
    for input, label in val_loader:
        input = cuda(input)
        label = cuda(label)
        pred = model(input)
        for i in range(len(input)):
            if Accuracy(pred[i], label[i]):
                correct += 1
    correct /= len(val_visit)
    print(correct, 'use time:', time.clock() - start_time)

    result = []
    num = 0
    for input, xxx in test_loader:
        preds = model(cuda(torch.tensor(input).float()))
        #print(num, input)
        oneres = 0
        #print(num, pred)
        for pred in preds:
            for i in range(len(pred)):
                if pred[i] > pred[oneres]:
                    oneres = i
            result.append(str(num).zfill(6) + '\t00' + str(oneres + 1))
            num += 1
    with open('data/results/' + str(epoch).zfill(5) + '_' + str(correct) +  '.txt', 'w') as f:
        f.write('\n'.join(result))
