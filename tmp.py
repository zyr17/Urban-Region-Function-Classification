import pickle
import numpy as np

def ndarray(filename):
    res = []

    with open(filename, 'rb') as f:
        res = pickle.load(f)

    #print(res[0])

    tot = 0
    for i in res:
        for j in i:
            tot += len(j[1])
            j[1] = np.array(j[1])

    #print(tot)

    pickle.dump(res, open(filename, 'wb'))

if __name__ == '__main__':
    #ndarray('train_visit_0_10000.pkl')
    ndarray('train_visit_10000_20000.pkl')
    ndarray('train_visit_20000_30000.pkl')
    ndarray('train_visit_30000_40000.pkl')