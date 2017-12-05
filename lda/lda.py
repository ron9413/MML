import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from pca_b import *

def getFisherFaces(trainingfaces):
    N = trainingfaces.shape[1]
    c = 40
    eigfaces, meanImage = getEigenfaces(trainingfaces)
    X = projectFaces(trainingfaces, eigfaces[:, :N-c], meanImage)
    Sw = np.zeros((N-c, N-c))
    Sb = np.zeros((N-c, N-c))
    for i in range(c):
        # calculate within-class scatter matrix
        Xw = X[:, 3*i:3*(i+1)]
        Xw_mean = np.mean(Xw, axis=1).reshape(-1, 1)
        Xw = Xw - Xw_mean
        Sw += np.matmul(Xw, Xw.T)
        # calculate between-class scatter matrix
        global_mean = np.mean(X, axis=1).reshape(-1, 1)
        Xb = Xw_mean - global_mean
        Sb += np.matmul(Xb, Xb.T)
    A = np.matmul(inv(Sw), Sb)
    eigVal, w = eigs(A, c-1)
    return w, eigfaces, meanImage

def projectFacesLDA(testingfaces, w, eigfaces, meanImage):
    dim = w.shape[0]
    testingCoefs_pca = projectFaces(testingfaces, eigfaces[:, :dim], meanImage)
    p = np.matmul(w.T, testingCoefs_pca)
    return p

if __name__ == "__main__":
    facedata = sio.loadmat("facedata.mat")["facedata"]
    trainingfaces, testingfaces = splitData(facedata)
    w, Eigenfaces, meanImage = getFisherFaces(trainingfaces)
    testingCoefs = projectFacesLDA(testingfaces, w, Eigenfaces, meanImage)
    trainingCoefs = projectFacesLDA(trainingfaces, w, Eigenfaces, meanImage)
    checkCluster(testingCoefs)
    IDRates = []
    for Mode in [1, 2, 3]:
        SimM = getSimM(trainingCoefs, testingCoefs, Mode)
        for rank in [1, 2, 3]:
            IDRates.append(getIdentificationRank(SimM, rank))
    IDRates = np.asarray(IDRates).reshape(3, 3)
    print(IDRates)
    plt.show()
