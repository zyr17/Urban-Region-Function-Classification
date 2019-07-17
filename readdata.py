import os
import pickle
from PIL import Image
import numpy as np
import time
import sys
import random
import re

DF = 'data/'
train_image_folder = DF + 'original/train_image/'
train_visit_folder = DF + 'original/train_visit/'
test_image_folder = DF + 'original/test_image/'
test_visit_folder = DF + 'original/test_visit/'

def read_train_data(start, end):
    tilist = os.listdir(train_image_folder)
    tvlist = os.listdir(train_visit_folder)
    tilist.sort()
    tvlist.sort()

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
    pickle.dump(train_image, open(DF + 'pickle/part/train_image_' + str(start) + '_' + str(end) + '.pkl', 'wb'))
    pickle.dump(train_visit, open(DF + 'pickle/part/train_visit_' + str(start) + '_' + str(end) + '.pkl', 'wb'))
    pickle.dump(train_label, open(DF + 'pickle/part/train_label_' + str(start) + '_' + str(end) + '.pkl', 'wb'))

def read_test_data(start = 0, end = 10000):
    tilist = os.listdir(test_image_folder)
    tvlist = os.listdir(test_visit_folder)
    tilist.sort()
    tvlist.sort()

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

def visit2bucket(filename, folder):
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
        with open(folder + filename[-17:], 'wb') as f2:
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
    quick = np.array([-1] * 10000000, dtype = 'int32')
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

def remove_short_expand(starti, visitname, labelname, outputname, istestdata = False, threshold = 23):
    with open(visitname, 'rb') as f:
        visits = pickle.load(f)
    labels = np.zeros(len(visits))
    if not istestdata:
        with open(labelname, 'rb') as f:
            labels = pickle.load(f)
    resvisit = []
    reslabel = []
    othervisit = []
    otherlabel = []
    for num in range(len(visits)):
        if num % 100 == 0:
            print(num, len(othervisit))
        visit = visits[num]
        label = labels[num]
        flag = 0
        for line in visit:
            if len(line) > threshold:
                if flag == 0:
                    othervisit.append(line)
                    otherlabel.append(starti + num)
                else:
                    resvisit.append(line)
                    reslabel.append(label)
                flag = 1 - flag
                if istestdata:
                    flag = 0
    reslabel = np.array(reslabel, dtype='int8')
    otherlabel = np.array(otherlabel, dtype='int32')
    print('save', outputname)
    if not istestdata:
        with open(outputname, 'wb') as f:
            pickle.dump([resvisit, reslabel], f)
    with open(outputname + '.1', 'wb') as f:
        pickle.dump([othervisit, otherlabel], f)

def combine(folder, start = 0, end = 400000, oldstep = 10000, newstep = 100000, suffix = ''):
    isarray = False
    for i in range(start, end, newstep):
        res = []
        savefile = '%sc_%06d_%06d.pkl%s' % (folder, i, i + newstep, suffix)
        for j in range(i, i + newstep, oldstep):
            filename = '%s%06d_%06d.pkl%s' % (folder, j, j + oldstep, suffix)
            print(filename)
            data = pickle.load(open(filename, 'rb'))
            isarray = isinstance(data, np.ndarray)
            if isarray:
                res.append(data)
            else:
                res += data
        if isarray:
            res = np.concatenate(res)
        print('save', savefile)
        pickle.dump(res, open(savefile, 'wb'))

if __name__ == '__main__':
    '''
    for i in range(0, 400000, 10000):
        read_train_data(i, i + 10000)
    for i in range(0, 100000, 10000):
        read_test_data(i, i + 10000)
    '''
    '''
    read_test_data()
    visit2bucket('test_visit_0_10000.pkl')
    '''
    ''' 
    #count uuid
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
    #visit2bucket
    #res = visit2bucket('pickle/train_visit_0_10000.pkl')
    #for i in res:
    #    for j in i:
    #        print(str(j) + ' ', end = '')
    #    print()
    #exit()
    for i in range(20000, 400000, 10000):
        print('i', i)
        folder = 'data/pickle/train_visit_bucket/'
        visit2bucket('data/pickle/train_visit/%06d_%06d.pkl' % (i, i + 10000), folder)
    for i in range(0, 100000, 10000):
        print('i', i)
        folder = 'data/pickle/test_visit_bucket/'
        visit2bucket('data/pickle/test_visit/%06d_%06d.pkl' % (i, i + 10000), folder)
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
    '''
    #with id to without id
    for i in range(250000, 400000, 10000):
        print('train', i)
        visit_data_expand('data/pickle/train_visit/%06d_%06d.pkl' % (i, i + 10000), 'data/pickle/train_visit_without_id/%06d_%06d.pkl' % (i, i + 10000))
    for i in range(0, 100000, 10000):
        print('test', i)
        visit_data_expand('data/pickle/test_visit/%06d_%06d.pkl' % (i, i + 10000), 'data/pickle/test_visit_without_id/%06d_%06d.pkl' % (i, i + 10000))
    '''
    #visit_data_expand('E:/BaiduNetdiskDownload/初赛赛题/pickle/test_visit_with_id.pkl', 'data/pickle/test_visit.pkl')
    #remove_short_expand('data/pickle/test_visit.pkl', 'data/pickle/test_label.pkl', 'test_visitline_23.pkl')
    filenames = [
        'data/pickle/part/train_visit_0_10000.pkl',
        'data/pickle/part/train_visit_10000_20000.pkl',
        'data/pickle/part/train_visit_20000_30000.pkl',
        'data/pickle/part/train_visit_30000_40000.pkl'
    ]

    #res = open('result.txt', 'w')
    '''
    #analyze num of ids for one input
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            for one in data:
                res.write(str(len(one)) + '\n')
    '''
    '''
    #analyze num of timestamp for one line
    bucket = [0] * (26 * 7 * 24)
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(filename)
            for num, one in enumerate(data):
                if (num % 1000 == 0):
                    print(num)
                for num in one:
                    bucket[len(num)] += 1
    res.write('\n'.join([str(x) for x in bucket]))
    '''
    '''
    #long visitline
    for i in range(0, 400000, 10000):
        visit = 'data/pickle/train_visit/%06d_%06d.pkl' % (i, i + 10000)
        label = 'data/pickle/train_label/%06d_%06d.pkl' % (i, i + 10000)
        res = 'data/pickle/visitline_train/%06d_%06d.pkl' % (i, i + 10000)
        remove_short_expand(i, visit, label, res)
    
    for i in range(0, 100000, 10000):
        visit = 'data/pickle/test_visit/%06d_%06d.pkl' % (i, i + 10000)
        label = 'data/pickle/test_label/%06d_%06d.pkl' % (i, i + 10000)
        res = 'data/pickle/visitline_test/%06d_%06d.pkl' % (i, i + 10000)
        remove_short_expand(i, visit, label, res, True)
    '''
    '''
    #combine vilitline
    totalv = []
    totall = []
    for i in range(0, 40000, 10000):
        with open('data/pickle/part/visitline_23/%d_%d.pkl.1' % (i, i + 10000), 'rb') as f:
            [visit, label] = pickle.load(f)
            print(i)
            for num in range(len(visit)):
                totalv.append(visit[num])
                totall.append(label[num])
    totall = np.array(totall, dtype='int32')
    pickle.dump([totalv, totall], open('data/pickle/train_visitline_23.pkl.1', 'wb'))
    '''
    '''
    #shuffle visitline
    for i in range(0, 400000, 100000):
        openfile = 'data/pickle/visitline_train/c_%06d_%06d.pkl' % (i, i + 100000)
        savefile = 'data/pickle/visitline_train_shuffle/c_%06d_%06d.pkl' % (i, i + 100000)
        print(openfile, savefile)
        [train_x, train_y] = pickle.load(open(openfile, 'rb'))
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
        pickle.dump([train_x, train_y], open(savefile, 'wb'))
    '''
    '''
    #visit length
    for i in range(0, 100000, 10000):
        filename = 'data/pickle/test_visit/%06d_%06d.pkl' % (i, i + 10000)
        savename = 'data/pickle/test_visit_length/%06d_%06d.pkl' % (i, i + 10000)
        print(filename)
        data = pickle.load(open(filename, 'rb'))
        length = [list(map(len, x)) for x in data]
        result = np.zeros((10000, 24 * 7 * 26), dtype = 'int32')
        for num, one in enumerate(length):
            for j in one:
                result[num][j] += 1
        pickle.dump(result, open(savename, 'wb'))
    '''
    '''
    folders = [x for x in os.listdir('data/pickle/') if not re.search('with_id|_visit$', x)]
    folders.sort()
    #print(folders)
    for folder in folders:
        end = 400000
        if folder[:4] == 'test':
            end = 100000
        print(folder, end)
        combine('data/pickle/' + folder + '/', 0, end)
    '''
    #combine('data/pickle/visitline_train/', 0, 100000)
    #combine('data/pickle/visitline_test/', 0, 100000)
    '''
    #combine visitline
    #combine('data/pickle/visitline_train/', 0, 400000)
    #combine('data/pickle/visitline_train_1/', 0, 400000)
    for i in range(0, 400000, 100000):
        openfile = 'data/pickle/visitline_train/c_%06d_%06d.pkl' % (i, i + 100000)
        savefile = 'data/pickle/visitline_train/d_%06d_%06d.pkl' % (i, i + 100000)
        data = pickle.load(open(openfile, 'rb'))
        X = data[::2]
        y = np.concatenate(data[1::2])
        x = []
        for xx in X:
            x += xx
        print(len(x), len(y))
        pickle.dump([x, y], open(savefile, 'wb'))
        openfile = 'data/pickle/visitline_train_1/c_%06d_%06d.pkl' % (i, i + 100000)
        savefile = 'data/pickle/visitline_train_1/d_%06d_%06d.pkl' % (i, i + 100000)
        data = pickle.load(open(openfile, 'rb'))
        X = data[::2]
        y = np.concatenate(data[1::2])
        x = []
        for xx in X:
            x += xx
        print(len(x), len(y))
        pickle.dump([x, y], open(savefile, 'wb'))
    '''