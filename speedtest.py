import numpy as np
import random
import time

a = []
for i in range(100):
    b = []
    for i in range(50):
        b.append(random.randint(0, 26 * 7 * 24 - 1))
    b = np.array(b, dtype='int16')
    a.append(b)

origin = time.time()
start = time.time()

for i in range(100000):
    if i % 100 == 0:
        print(i, time.time() - start)
        start = time.time()
    arr = []
    for num, aa in enumerate(a):
        arr2 = np.zeros(26 * 7 * 24)
        for j in aa:
            arr2[j] = 1
        arr.append(arr2)
    arr = np.concatenate(arr)

print('end', time.time() - origin)