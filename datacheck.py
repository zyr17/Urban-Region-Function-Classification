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
def r_check_label(data):
    data = np.array(data)
    return np.sum(data, axis = 0)

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

#res = pklenum('data/pickle/train_label/', m_check_label, r_check_label)
#res = pklenum('data/pickle/train_label/', numpy_pack)
res = pklenum('data/pickle/train_visit/', m_idnum_in_pos, r_concat)

#print(res)
open('output.txt', 'w').write('\n'.join([str(x) for x in res]))