import pickle
import torch
import torchvision
import numpy as np
import time

torch.manual_seed(19951017)
label_num = 9

with open('data/pickle/train_visit.bucket.pkl', 'rb') as f:
    train_visit = pickle.load(f)
with open('data/pickle/train_visit.length.pkl', 'rb') as f:
    train_length = pickle.load(f)
with open('data/pickle/train_label.pkl', 'rb') as f:
    train_label_raw = pickle.load(f)
train_label = np.zeros((len(train_label_raw), label_num), dtype='float')
for num in range(len(train_label_raw)):
    train_label[num][train_label_raw[num] - 1] = 1
with open('data/pickle/train_image.pkl', 'rb') as f:
    train_image = pickle.load(f)

with open('data/pickle/test_visit.bucket.pkl', 'rb') as f:
    test_visit = pickle.load(f)
with open('data/pickle/test_visit.length.pkl', 'rb') as f:
    test_length = pickle.load(f)
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
train_length = train_length[:200]
test_visit = test_visit[:100]
test_image = test_image[:100]
test_length = test_length[:100]
'''
# use last 10% data as validation set
val_line = len(train_visit) // 10 * 9
val_visit = train_visit[val_line:]
val_label = train_label[val_line:]
val_image = train_image[val_line:]
val_length = train_length[val_line:]
train_visit = train_visit[:val_line]
train_label = train_label[:val_line]
train_image = train_image[:val_line]
train_length = train_length[:val_line]

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

#feature_length, kernel_size, padding, maxpool_size, maxpool_padding
image_input = (3, 100, 100)
image_CNN = (
    (50, 5, 2, 2, (0, 0)),
    (200, 5, 2, 2, (0, 0)),
    (400, 3, 1, 3, (1, 1)),
    (800, 3, 1, 3, (0, 0)),
)
image_FC = (800 * 3 * 3, 240)

visit_input = (7, 26, 24)
visit_CNN = (
    (100, 5, 2, 2, (0, 0)),
    (200, 5, 2, 2, (1, 0)),
    (400, 5, 2, 2, (1, 0)),
    (800, 5, 2, 2, (0, 1)),
)
visit_FC = (800 * 2 * 2, 160)

length_input = (26 * 7 * 24,)
length_CNN = ()
length_FC = (26 * 7 * 24, 1000, 200)

line_input = (500 * (1 + label_num),) # at most 500 line results. length + label. randomly select, TODO: can repeat?
line_CNN = ()
line_FC = (500 * (1 + label_num), 1000, 200)

class CNNpart(torch.nn.Module):
    def __init__(self, inputlen, CNN, FC):
        super(CNNpart, self).__init__()
        self.CNN = CNN
        self.FC = FC
        self.visit_convs = torch.nn.ModuleList()
        self.visit_fcs = torch.nn.ModuleList()
        lastfea = inputlen[0]
        for i in CNN:
            self.visit_convs.append(torch.nn.Sequential(
                torch.nn.Conv2d(lastfea, i[0], i[1], padding=i[2]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(i[3], padding=i[4])
            ))
            lastfea = i[0]
        lastfea = FC[0]
        for i in FC[1:]:
            self.visit_fcs.append(torch.nn.Sequential(
                torch.nn.Linear(lastfea, i),
                torch.nn.ReLU()
            ))
            lastfea = i
    def forward(self, inputs):
        x = inputs
        for conv in self.visit_convs:
            x = conv(x)
        x = x.reshape(-1, self.FC[0])
        for fc in self.visit_fcs:
            x = fc(x)
        return x

class CNN(torch.nn.Module):
    def __init__(self, inputlen = (7, 26, 24), outputlen = label_num):
        super(CNN, self).__init__()
        self.CNNpart = CNNpart(visit_input, visit_CNN, visit_FC)
        self.fc.append(torch.nn.Sequential(
            torch.nn.Linear(160, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images):
        x = inputs # [B, 7, 26, 24]
        x = self.CNNpart(x)
        x = self.fc(x)
        return x

#TODO: MaxPool2d在不整除时行为
class concat_after(torch.nn.Module):
    def __init__(self, outputlen = label_num):
        super(concat_after, self).__init__()
        self.imageCNN = CNNpart(image_input, image_CNN, image_FC)
        self.visitCNN = CNNpart(visit_input, visit_CNN, visit_FC)
        self.lengthCNN = CNNpart(length_input, length_CNN, length_FC)

        self.final_fcs = torch.nn.ModuleList()
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(image_FC[-1] + visit_FC[-1] + length_FC[-1], 80),
            torch.nn.ReLU()
        ))# [B, 80]
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(80, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images, length):
        x = inputs # [B, 7, 26, 24]
        x = self.visitCNN(x)
        
        y = images
        y = self.imageCNN(y)

        z = length
        z = self.lengthCNN(z)
        
        zz = torch.cat([x, y, z], 1)
        for fc in self.final_fcs:
            zz = fc(zz)
        
        return zz

class VisitDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, x3, y):
        self.x1 = torch.tensor(x1).float()
        self.x2 = torch.tensor(x2).float()
        self.x3 = torch.tensor(x3).float()
        self.y = torch.tensor(y).float()
    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.x3[index], self.y[index]
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
train_dataset = VisitDataset(train_visit, train_image, train_length, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
test_dataset = VisitDataset(test_visit, test_image, test_length, np.zeros(len(test_visit)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, False)
val_dataset = VisitDataset(val_visit, val_image, val_length, val_label)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, True)
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
    for input, image, length, label in train_loader:
        input = cuda(input)
        image = cuda(image)
        length = cuda(length)
        label = cuda(label)
        opt.zero_grad()
        pred = model(input, image, length)
        L = loss(pred, label)
        if (batch_count % (len(train_label) // 10 // batch_size) == 0):
            print(batch_count, L.data.item())
        L.backward()
        opt.step()
        batch_count += 1
    model.eval()
    correct = 0
    for input, image, length, label in val_loader:
        input = cuda(input)
        image = cuda(image)
        length = cuda(length)
        label = cuda(label)
        pred = model(input, image, length)
        for i in range(len(input)):
            if Accuracy(pred[i], label[i]):
                correct += 1
    correct /= len(val_visit)
    print(correct, 'use time:', time.clock() - start_time)

    result = []
    num = 0
    for input, image, length, xxx in test_loader:
        preds = model(cuda(torch.tensor(input).float()), cuda(torch.tensor(image).float()), cuda(torch.tensor(length).float()))
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
