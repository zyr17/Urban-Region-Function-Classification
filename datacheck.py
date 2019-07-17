import pickle
import numpy as np

def pklenum(folder, mfunc = lambda x:x, rfunc = lambda x:x, start = 0, end = 400000):
    res = []
    for i in range(start, end, 10000):
        filename = '%s%06d_%06d.pkl' % (folder, i, i + 10000)
        print(filename)
        data = pickle.load(open(filename, 'rb'))
        res.append(mfunc(data, folder, i))
    return rfunc(res)

def m_check_label(data, folder, index):
    res = [0] * 10
    data = np.array(data)
    for i in range(0, 10):
        res[i] = np.sum(data == i)
    return res

def numpy_pack(data, folder, index):
    file = 'data/pickle/temp/%06d_%06d.pkl' % (index, index + 10000)
    data = np.array(data)
    print(data.shape, data.dtype)
    pickle.dump(data, open(file, 'wb'))
    return None

def m_idnum_in_pos(data, folder, index):
    return np.array(list(map(len, data)))

def r_concat(data):
    return np.concatenate(data)
def r_add(data):
    return np.sum(np.array(data), axis = 0)

def m_length_in_label(data, folder, index):
    nowlabel = labels[index:index + 10000]
    result = np.zeros((10, 26 * 24 * 7), dtype = 'int32')
    for one, l in zip(data, nowlabel):
        result[l] += one
    return result

#res = pklenum('data/pickle/train_label/', m_check_label, r_add)
#res = pklenum('data/pickle/train_label/', numpy_pack)
#res = pklenum('data/pickle/train_visit/', m_idnum_in_pos, r_concat)
labels = pklenum('data/pickle/train_label/', lambda x:x, r_concat)
res = pklenum('data/pickle/train_visit_length/', m_length_in_label, r_add)

#print(res)
open('output.txt', 'w').write('\n'.join([str(x) for x in res]))