#https://github.com/pfchai/Haze-Removal

from PIL import Image
import numpy as np
import pickle

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q

class HazeRemoval:
    def __init__(self, omega = 0.85, r = 40):
        self.omega = omega
        self.r = r
        self.eps = 10 ** (-3)
        self.t = 0.1

    def _ind2sub(self, array_shape, ind):
        rows = (ind.astype('int') // array_shape[1])
        cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
        return (rows, cols)

    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def haze_removal(self, imagearr):
        oriImage = imagearr
        img = np.array(oriImage).astype(np.double) / 255.0
        grayImage = self._rgb2gray(img)

        darkImage = img.min(axis=2)

        (i, j) = self._ind2sub(darkImage.shape, darkImage.argmax())
        A = img[i, j, :].mean()
        transmission = 1 - self.omega * darkImage / A

        transmissionFilter = guidedfilter(grayImage, transmission, self.r, self.eps )
        transmissionFilter[transmissionFilter < self.t] = self.t

        resultImage = np.zeros_like(img)
        for i in range(3):
            resultImage[:, :, i] = (img[:, :, i] - A) / transmissionFilter + A

        resultImage[resultImage < 0] = 0
        resultImage[resultImage > 1] = 1
        result = (resultImage * 255).astype(np.uint8)

        return result

if __name__ == '__main__':
    pklname = 'data/pickle/test_image'
    data = pickle.load(open(pklname + '.pkl', 'rb'))
    hz = HazeRemoval()
    res = np.zeros(data.shape, dtype='uint8')

    for num, i in enumerate(data):
        result = hz.haze_removal(i)
        res[num,:,:,:] = result
        #Image.fromarray(result).show()
        #input()
        if num % 100 == 0:
            print(num)
    pickle.dump(res, open(pklname + '.haze.pkl', 'wb'))