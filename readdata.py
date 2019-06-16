import os
import pickle
from PIL import Image
import numpy as np
import time
import sys

DF = 'data/'
train_image_folder = DF + 'train_image/'
train_visit_folder = DF + 'train_visit/'
test_image_folder = DF + 'test_image/'
test_visit_folder = DF + 'test_visit/'

def read_train_data(start, end):
    tilist = os.listdir(train_image_folder)
    tvlist = os.listdir(train_visit_folder)

    train_label = []
    train_image = []
    train_visit = []

    tilist = tilist[start:end]
    tvlist = tvlist[start:end]

    for i in range(len(tilist)):
        if i % 100 == 0:
            print('processing No. ' + str(i) + ', total: ' + str(len(tilist)))
        label = int(tilist[i][9])
        pic = np.asarray(Image.open(train_image_folder + tilist[i]))
        visit = []
        with open(train_visit_folder + tvlist[i]) as f:
            for line in f.readlines():
                line = line.strip()
                id = line[:16]
                data = []
                for oneday in line[17:].split(','):
                    day = oneday[:8]
                    for hr in oneday[9:].split('|'):
                        data.append(int(day + hr))
                visit.append([id, np.array(data)])
        train_label.append(label)
        train_image.append(pic)
        train_visit.append(visit)

    #print(train_image, train_visit, train_label)
    
    print('saving...')
    pickle.dump(train_image, open('train_image_' + str(start) + '_' + str(end) + '.pkl', 'wb'))
    pickle.dump(train_visit, open('train_visit_' + str(start) + '_' + str(end) + '.pkl', 'wb'))
    pickle.dump(train_label, open('train_label_' + str(start) + '_' + str(end) + '.pkl', 'wb'))

def read_test_data(start = 0, end = 10000):
    tilist = os.listdir(test_image_folder)
    tvlist = os.listdir(test_visit_folder)

    test_image = []
    test_visit = []

    tilist = tilist[start:end]
    tvlist = tvlist[start:end]

    for i in range(len(tilist)):
        if i % 100 == 0:
            print('processing No. ' + str(i) + ', total: ' + str(len(tilist)))
        pic = np.asarray(Image.open(test_image_folder + tilist[i]))
        visit = []
        with open(test_visit_folder + tvlist[i]) as f:
            for line in f.readlines():
                line = line.strip()
                id = line[:16]
                data = []
                for oneday in line[17:].split(','):
                    day = oneday[:8]
                    for hr in oneday[9:].split('|'):
                        data.append(int(day + hr))
                visit.append([id, np.array(data)])
        test_image.append(pic)
        test_visit.append(visit)

    #print(test_image, test_visit, test_label)
    
    print('saving...')
    pickle.dump(test_image, open('test_image_' + str(start) + '_' + str(end) + '.pkl', 'wb'))
    pickle.dump(test_visit, open('test_visit_' + str(start) + '_' + str(end) + '.pkl', 'wb'))

def getids(filename):
    data = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    res = []
    for i in data:
        for j in i:
            res.append(j[0])
    return res

def visit2bucket(filename):
    '''
    #min: 2018100100 max: 2019033100
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        min = 2100010101
        max = 0
        for i in data:
            for j in i:
                for k in j[1]:
                    if k > max:
                        max = k
                    if k < min:
                        min = k
    print(min, max)
    '''
    with open(filename, 'rb') as f:
        quick = np.array([-1] * 10000000, dtype = 'int32')
        data = pickle.load(f)
        res = np.zeros((len(data), 26 * 7 * 24), dtype = 'int32')
        starttime = time.mktime(time.strptime('2018100100', '%Y%m%d%H'))
        for num, i in enumerate(data):
            if num % 10 == 0:
                sys.stderr.write(str(num) + '\n')
            for j in i:
                for k in j[1]:
                    nowtime = quick[k - 2010000000]
                    if nowtime == -1:
                        nowtime = time.mktime(time.strptime(str(k), '%Y%m%d%H'))
                        quick[k - 2010000000] = nowtime
                    pos = int((nowtime - starttime) / 60 / 60)
                    res[num][pos] += 1
        with open(filename[:-4] + '.bucket.pkl', 'wb') as f2:
            pickle.dump(res, f2)
        return res

def concat(filenames, outputname):
    arrays = []
    for name in filenames:
        arrays.append(pickle.load(open(name, 'rb')))
    res = np.concatenate(arrays)
    print(res.shape)
    pickle.dump(res, open(outputname, 'wb'))

def visit_data_expand(filename, outputname):
    with open(filename, 'rb') as f:
        visit = pickle.load(f)
    res = []
    quick = np.array([-1] * 10000000, dtype = 'int16')
    starttime = time.mktime(time.strptime('2018100100', '%Y%m%d%H'))
    for num, one in enumerate(visit):
        resone = []
        for line in one:
            line = line[1]
            resline = []
            for onetime in line:
                nowtime = quick[onetime - 2010000000]
                if nowtime == -1:
                    nowtime = time.mktime(time.strptime(str(onetime), '%Y%m%d%H'))
                    quick[onetime - 2010000000] = nowtime
                pos = int((nowtime - starttime) / 60 / 60)
                resline.append(pos)
            resline = np.array(resline, dtype='int16')
            resone.append(resline)
        res.append(resone)
        print(num)
    pickle.dump(res, open(outputname, 'wb'))

if __name__ == '__main__':
    ''' 
    read_train_data(0, 10000)
    read_train_data(10000, 20000)
    read_train_data(20000, 30000)
    read_train_data(30000, 40000)
    '''
    '''
    read_test_data()
    visit2bucket('test_visit_0_10000.pkl')
    '''
    ''' 
    res = []
    for i in getids('pickle/train_visit_0_10000.pkl'):
        res.append(i)
    print('1 complete')
    for i in getids('pickle/train_visit_10000_20000.pkl'):
        res.append(i)
    print('2 complete')
    for i in getids('pickle/train_visit_20000_30000.pkl'):
        res.append(i)
    print('3 complete')
    for i in getids('pickle/train_visit_30000_40000.pkl'):
        res.append(i)
    print('4 complete')
    print(len(res))
    resint = []
    for i in res:
        resint.append(int(i, 16))
    res = resint
    m = {}
    for i in res:
        m[i] = 0
    for i in res:
        m[i] += 1
    print(len(m.keys()))
    with open('out.txt', 'w') as f:
        f.write(str(m))
    '''

    '''
    res = visit2bucket('pickle/train_visit_0_10000.pkl')
    #for i in res:
    #    for j in i:
    #        print(str(j) + ' ', end = '')
    #    print()
    #exit()
    res = visit2bucket('pickle/train_visit_10000_20000.pkl')
    res = visit2bucket('pickle/train_visit_20000_30000.pkl')
    res = visit2bucket('pickle/train_visit_30000_40000.pkl')
    '''
    '''
    concat([
        'data/pickle/train_label_0_10000.pkl',
        'data/pickle/train_label_10000_20000.pkl',
        'data/pickle/train_label_20000_30000.pkl',
        'data/pickle/train_label_30000_40000.pkl',
    ], 'train_label.pkl')
    concat([
        'data/pickle/train_image_0_10000.pkl',
        'data/pickle/train_image_10000_20000.pkl',
        'data/pickle/train_image_20000_30000.pkl',
        'data/pickle/train_image_30000_40000.pkl',
    ], 'train_image.pkl')
    concat([
        'data/pickle/train_visit_0_10000.bucket.pkl',
        'data/pickle/train_visit_10000_20000.bucket.pkl',
        'data/pickle/train_visit_20000_30000.bucket.pkl',
        'data/pickle/train_visit_30000_40000.bucket.pkl',
    ], 'train_visit.bucket.pkl')
    '''
    visit_data_expand('data/pickle/part/train_visit_with_id_0_10000.pkl', 'data/pickle/part/train_visit_0_10000.pkl')
    visit_data_expand('data/pickle/part/train_visit_with_id_10000_20000.pkl', 'data/pickle/part/train_visit_10000_20000.pkl')
    #visit_data_expand('data/pickle/part/train_visit_with_id_20000_30000.pkl', 'data/pickle/part/train_visit_20000_30000.pkl')
    #visit_data_expand('data/pickle/part/train_visit_with_id_30000_40000.pkl', 'data/pickle/part/train_visit_30000_40000.pkl')