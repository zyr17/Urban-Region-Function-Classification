import pickle
import torch
import torchvision
import numpy as np
import time
import pdb
import dpn
import random
import os

torch.manual_seed(19951017)
random.seed(19951017)
label_num = 9
"""
with open('data/pickle/train_visit.bucket.pkl', 'rb') as f:
    train_visit = pickle.load(f)
with open('data/pickle/train_visit.length.pkl', 'rb') as f:
    train_length = pickle.load(f)
with open('data/pickle/train_label.pkl', 'rb') as f:
    train_label_raw = pickle.load(f)
'''
train_label = np.zeros((len(train_label_raw), label_num), dtype='float')
for num in range(len(train_label_raw)):
    train_label[num][train_label_raw[num] - 1] = 1
'''
train_label = train_label_raw
train_label -= 1
with open('data/pickle/train_image.pkl', 'rb') as f:
    train_image = pickle.load(f)
with open('data/pickle/train_visitline_res.pkl', 'rb') as f:
    train_visitline = pickle.load(f)

print('load train data done')

with open('data/pickle/test_visit.bucket.pkl', 'rb') as f:
    test_visit = pickle.load(f)
with open('data/pickle/test_visit.length.pkl', 'rb') as f:
    test_length = pickle.load(f)
with open('data/pickle/test_image.pkl', 'rb') as f:
    test_image = pickle.load(f)
with open('data/pickle/test_visitline_res.pkl', 'rb') as f:
    test_visitline = pickle.load(f)

print('load test data done')

for i in train_visitline:
    if (len(i)) > 0:
        i[:,0] = 0
for i in test_visitline:
    if (len(i)) > 0:
        i[:,0] = 0

def change_visit_bucket_shape(visit):
    visit = np.array(visit, dtype='int32')
    visit = visit.reshape(-1, 26, 7, 24)
    visit = visit.transpose(0, 2, 1, 3)
    visit = visit.reshape(-1, 26 * 7 * 24)
    return visit

train_visit = change_visit_shape(train_visit)
test_visit = change_visit_shape(test_visit)
train_image = train_image.transpose(0, 3, 1, 2)
test_image = test_image.transpose(0, 3, 1, 2)
# use last 10% data as validation set TODO: random_split
val_line = len(train_visit) // 10 * 9
val_visit = train_visit[val_line:]
val_label = train_label[val_line:]
val_image = train_image[val_line:]
val_length = train_length[val_line:]
val_visitline = train_visitline[val_line:]
train_visit = train_visit[:val_line]
train_label = train_label[:val_line]
train_image = train_image[:val_line]
train_length = train_length[:val_line]
train_visitline = train_visitline[:val_line]
"""
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

visitline_max = 500
visitline_input = (visitline_max * (1 + label_num),) # at most 500 line results. length + label. randomly select, TODO: can repeat?
visitline_CNN = ()
visitline_FC = (visitline_max * (1 + label_num), 1000, 200)

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
                #torch.nn.Dropout(0.5),
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
        self.fc = (torch.nn.Sequential(
            torch.nn.Linear(160, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images, length, visitline):
        x = inputs # [B, 7, 26, 24]
        x = self.CNNpart(x)
        x = self.fc(x)
        return x

#TODO: MaxPool2d在不整除时行为
class concat_after(torch.nn.Module):
    def __init__(self, outputlen = label_num):
        super(concat_after, self).__init__()
        self.imageCNN = CNNpart(image_input, image_CNN, image_FC)
        #self.visitCNN = CNNpart(visit_input, visit_CNN, visit_FC)
        self.visitCNN = dpn.DPN26()
        self.lengthCNN = CNNpart(length_input, length_CNN, length_FC)
        self.visitlineCNN = CNNpart(visitline_input, visitline_CNN, visitline_FC)

        self.imageNorm = torch.nn.BatchNorm2d(3)
        self.visitNorm = torch.nn.BatchNorm2d(7)

        self.final_fcs = torch.nn.ModuleList()
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(image_FC[-1] + 256 + length_FC[-1] + visitline_FC[-1], 120),
            torch.nn.ReLU()
        ))# [B, 120]
        self.final_fcs.append(torch.nn.Sequential(
            torch.nn.Linear(120, outputlen),
            torch.nn.Sigmoid()
        ))# [B, 9]
    def forward(self, inputs, images, length, visitline):
        x = self.visitNorm(inputs) # [B, 7, 26, 24]
        x = self.visitCNN(x)
        #print(x.size())
        
        y = self.imageNorm(images)
        y = self.imageCNN(y)

        z = length
        z = self.lengthCNN(z)

        vl = visitline
        vl = self.visitlineCNN(vl)
        
        zz = torch.cat([x, y, z, vl], 1)
        for fc in self.final_fcs:
            zz = fc(zz)
        
        return zz

class VisitLineDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        if (len(x)) == 0:
            #add all zero element
            x = np.zeros((1, 1 + label_num), dtype='float')
        self.x = torch.tensor(x).float()
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

def random_transpose(img, rand = -1):
    #img = np.array, [3, N, N]
    if rand == -1:
        rand = random.randint(0,7)
    p1 = (rand & 1) + 1
    p2 = ((rand >> 1) & 1) * 2 - 1
    p3 = (rand >> 2) * 2 - 1
    return np.array(img.transpose(0, p1, 3 - p1)[:,::p2,::p3])

class VisitDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, x3, x4, y):
        self.x1 = torch.tensor(x1).float()#bucket
        self.x2 = x2#image
        self.x3 = torch.tensor(x3).float()#length
        self.x4loader = []#visitline
        self.x4iter = []
        for i in x4:
            visitline_dataset = VisitLineDataset(i)
            visitline_loader = torch.utils.data.DataLoader(visitline_dataset, visitline_max, True)
            self.x4loader.append(visitline_loader)
            self.x4iter.append(iter(self.x4loader[-1]))
        self.y = torch.tensor(y).long()
    def __getitem__(self, index):
        try:
            x4data = next(self.x4iter[index])
        except StopIteration:
            self.x4iter[index] = iter(self.x4loader[index])
            x4data = next(self.x4iter[index])
        x4 = np.zeros((visitline_max, 1 + label_num), dtype='float')
        #print(len(x4data))
        x4[:x4data.shape[0]] = x4data
        x4.reshape(visitline_max * (1 + label_num))
        x4 = torch.tensor(x4).float()
        x2 = torch.tensor(random_transpose(self.x2[index])).float()
        return self.x1[index], x2, self.x3[index], x4, self.y[index]
    def __len__(self):
        return len(self.x1)

def change_bucket_shape(bucket):
    bucket = np.array(bucket, dtype='int32')
    bucket = bucket.reshape(-1, 26, 7, 24)
    bucket = bucket.transpose(0, 2, 1, 3)
    bucket = bucket.reshape(-1, 26 * 7 * 24)
    return bucket

def read_one_data(folder, filenames, val_part = 0.01, read_front = False):
    res = []
    for filename in filenames:
        #print(folder + filename)
        data = pickle.load(open(folder + filename, 'rb'))
        val_line = int(len(data) * (1 - val_part))
        if read_front:
            data = data[:val_line]
        else:
            data = data[val_line:]
        if isinstance(data, list):
            res += data
        else:
            res.append(data)
    if len(res) == len(filenames):
        res = np.concatenate(res)
    return res

def norm_by_line(data):
    res = np.array(data, 'float')
    for i in res:
        i /= i.max()
    return res

def read_val_data(folder, buckets, lengths, labels, images, visitlines, savename, val_part = 0.01):
    if os.path.exists(savename):
        bucket, length, label, image, visitline = pickle.load(open(savename, 'rb'))
    else:
        bucket = read_one_data(folder, buckets, val_part)
        length = read_one_data(folder, lengths, val_part)
        label = read_one_data(folder, labels, val_part)
        image = read_one_data(folder, images, val_part)
        visitline = read_one_data(folder, visitlines, val_part)

        for i in visitline:
            if len(i) > 0:
                i[:,0] = 0
        bucket = change_bucket_shape(bucket)
        bucket = bucket.reshape(-1, 7, 26, 24)
        image = image.transpose(0, 3, 1, 2)
        label -= 1

        #norm
        length = norm_by_line(length)
        #bucket = norm_by_line(bucket)

        datas = [bucket, length, label, image]
        print(list(map(len, datas)), len(visitline), [x.shape for x in datas])
        pickle.dump([*datas, visitline], open(savename, 'wb'))
    dataset = VisitDataset(bucket, image, length, visitline, label)
    return torch.utils.data.DataLoader(dataset, 100, False)

def read_data(folder, bucketf, lengthf, labelf, imagef, visitlinef, batch_size, val_part = 0.01, is_shuffle = True):
    
    print('read data ... ')

    start_time = time.time()

    bucket = read_one_data(folder, [bucketf], val_part, True)
    length = read_one_data(folder, [lengthf], val_part, True)
    label = read_one_data(folder, [labelf], val_part, True)
    image = read_one_data(folder, [imagef], val_part, True)
    visitline = read_one_data(folder, [visitlinef], val_part, True)

    for i in visitline:
        if len(i) > 0:
            i[:,0] = 0
    bucket = change_bucket_shape(bucket)
    bucket = bucket.reshape(-1, 7, 26, 24)
    image = image.transpose(0, 3, 1, 2)
    label -= 1

    #norm
    length = norm_by_line(length)
    #bucket = norm_by_line(bucket)

    datas = [bucket, length, label, image]
    #print(list(map(len, datas)), len(visitline), [x.shape for x in datas])
    dataset = VisitDataset(bucket, image, length, visitline, label)

    print('read data over, time:', time.time() - start_time)

    return torch.utils.data.DataLoader(dataset, batch_size, is_shuffle)

picklefolder = 'data/pickle/'

c400000 = ['c_%06d_%06d.pkl' % (x, x + 100000) for x in range(0, 400000, 100000)]
train_buckets = ['train_visit_bucket/' + x for x in c400000]
train_lengths = ['train_visit_length/' + x for x in c400000]
train_labels = ['train_label/' + x for x in c400000]
train_images = ['train_image/' + x for x in c400000]
train_visitlines = ['visitline_train_res/' + x for x in c400000]
train_datas = list(zip(train_buckets, train_lengths, train_labels, train_images, train_visitlines))

c100000 = 'c_000000_100000.pkl'
test_bucket = 'test_visit_bucket/' + c100000
test_length = 'test_visit_length/' + c100000
test_label = 'test_label/' + c100000
test_image = 'test_image/' + c100000
test_visitline = 'visitline_test_res/' + c100000

batch_size = 64 
epoch_number = 1000
learning_rate = 0.0001
#modelname = 'CNN'
modelname = 'concat_after_norm_length'
savename = 'transpose'
#savename = 'visit_image'
save_count = 0
last_epoch = 0
save_interval = 1800
batch_interval = 120
result_interval = 30
nowtimestr = time.strftime('%d_%H%M%S', time.localtime())

"""
if (modelname != 'FC'):
    train_visit = train_visit.reshape(-1, 7, 26, 24)
    val_visit = val_visit.reshape(-1, 7, 26, 24)
    test_visit = test_visit.reshape(-1, 7, 26, 24)
train_dataset = VisitDataset(train_visit, train_image, train_length, train_visitline, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True)
test_dataset = VisitDataset(test_visit, test_image, test_length, test_visitline, np.zeros(len(test_visit)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, False)
val_dataset = VisitDataset(val_visit, val_image, val_length, val_visitline, val_label)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, True)
"""

val_loader = read_val_data(picklefolder, train_buckets, train_lengths, train_labels, train_images, train_visitlines, 'data/pickle/val_data.pkl')
test_loader = read_data(picklefolder, test_bucket, test_length, test_label, test_image, test_visitline, batch_size, 0, False)

if modelname == 'CNN':
    model = CNN()
elif modelname == 'FC':
    model = FC()
elif modelname == 'concat_after':
    model = concat_after()
model = cuda(model)
loss = torch.nn.CrossEntropyLoss()

'''
#load saved model
save_count = 0
last_epoch = 0
nowtimestr = '16_230218'
modelfolder = 'data/models/%s_%s/' % (savename, nowtimestr)
train_datas = train_datas[last_epoch:] + train_datas[:last_epoch]
model.load_state_dict(torch.load('%s%04d.model' % (modelfolder, save_count)))
'''

modelfolder = 'data/models/%s_%s/' % (savename, nowtimestr)
resultfolder = 'data/results/%s_%s/' % (savename, nowtimestr)
if not os.path.exists(modelfolder):
    os.mkdir(modelfolder)
    open(modelfolder + 'detail.txt', 'w').write('  id epoc    batch  correct\n')
if not os.path.exists(resultfolder):
    os.mkdir(resultfolder)

last_save_time = time.time()
last_batch_time = time.time()
print('start training')
for epoch in range(last_epoch, epoch_number):
    print('epoch', epoch)
    start_time = time.clock()
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    batch_count = 0

    if epoch % 4 == last_epoch % 4:
        train_loader = read_data(picklefolder, *train_datas[0], batch_size)
        train_datas = train_datas[1:] + train_datas[:1]

    for input, image, length, visitline, label in train_loader:
        model.train()
        input = cuda(input)
        image = cuda(image)
        length = cuda(length)
        visitline = cuda(visitline)
        label = cuda(label)
        opt.zero_grad()
        pred = model(input, image, length, visitline)
        L = loss(pred, label)
        if (time.time() - last_batch_time > batch_interval):
            print(batch_count, L.data.item())
            last_batch_time = time.time()
        L.backward()
        opt.step()
        batch_count += 1

        if time.time() - last_save_time > save_interval:
            #TODO: need record val data
            model.eval()
            correct = 0
            val_number = 0
            for input, image, length, visitline, label in val_loader:
                input = cuda(input)
                image = cuda(image)
                length = cuda(length)
                visitline = cuda(visitline)
                label = label
                pred = model(input, image, length, visitline)
                val_number += len(input)
                '''
                temp = pred.cpu().detach().numpy()
                temp = np.argmax(temp, axis = 1)
                print(type(temp), type(label), type(label.numpy()))
                print(temp.shape, label.shape)
                temp = temp == label.numpy()
                temp = np.sum(temp)
                temp = temp.numpy()
                '''
                correct += np.sum((np.argmax(pred.cpu().detach().numpy(), axis = 1) == label.numpy()))
            correct /= val_number
            print(correct, 'use time:', time.clock() - start_time)

            torch.save(model.state_dict(), modelfolder + '%04d.model' % (save_count,))
            open(modelfolder + 'detail.txt', 'a').write('%04d %04d %08d %.6f\n' % (save_count, epoch, batch_count, correct))
            
            result = []
            result_num = []
            num = 0
            print('start calc result')
            result_start_time = time.time()
            result_time = time.time()
            result_count = 0
            for input, image, length, visitline, xxx in test_loader:
                result_count += len(input)
                preds = model(cuda(torch.tensor(input).float()), cuda(torch.tensor(image).float()), cuda(torch.tensor(length).float()), cuda(torch.tensor(visitline).float()))
                #print(num, input)
                oneres = 0
                #print(num, pred)

                if time.time() - result_time > result_interval:
                    result_time = time.time()
                    print(result_count)

                result_num.append(np.argmax(preds.cpu().detach().numpy(), axis = 1))

            result_num = np.concatenate(result_num)
            result = ['%06d\t%03d' % (x[0], x[1] + 1) for x in zip(range(100000), result_num)]
            with open(resultfolder + str(epoch).zfill(4) + '.txt', 'w') as f:
                f.write('\n'.join(result))
            print('calc result over, time:', time.time() - result_time)
            last_save_time = time.time()
            save_count += 1

