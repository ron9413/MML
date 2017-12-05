import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.sparse.linalg import eigs
import time

def splitData(facedata):
    trainingfaces = np.stack(np.hstack(facedata[:, :3]))
    trainingfaces = trainingfaces.reshape(trainingfaces.shape[0], -1).T
    testingfaces = np.stack(np.hstack(facedata[:, 3:]))
    testingfaces = testingfaces.reshape(testingfaces.shape[0], -1).T
    return trainingfaces, testingfaces

def getEigenfaces(trainingfaces):
    N = trainingfaces.shape[1]
    mean = np.mean(trainingfaces, axis=1).reshape(-1, 1)
    X = trainingfaces - mean
    cov_matrix = np.matmul(X, X.T) / (N - 1)
    eigVal, eigVec = eigs(cov_matrix, N-1)
    idx = eigVal.astype(float).argsort()[::-1]
    eigVal = eigVal[idx].astype(float)
    eigVec = eigVec[:, idx].astype(float)
    return eigVec, mean

def projectFaces(testingfaces, eigfaces, meanImage):
    p = np.matmul(eigfaces.T, (testingfaces - meanImage))
    return p

def checkCluster(testingCoefs):
    num_class = 6
    colors = ['b', 'g', 'r', 'c', 'y', 'k']
    markers = ['o', '^', 's', '*', 'x', 'D']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_class):
        x = testingCoefs[0, 7*i:7*(i+1)]
        y = testingCoefs[1, 7*i:7*(i+1)]
        z = testingCoefs[2, 7*i:7*(i+1)]
        ax.scatter(x, y, z, c=colors[i], marker=markers[i])

def euclidean_dist(u, v):
    return norm(u - v, axis=0)

def manhattan_dist(u, v):
    return np.sum(np.abs(u - v), axis=0)

def cosine_dist(u, v):
    return 1 - np.matmul(u.T, v) / (norm(u, axis=0).reshape(-1, 1)*norm(v, axis=0).reshape(1, -1))

def getSimM(trainingCoefs, testingCoefs, Mode):
    dist = []
    if Mode == 1:
        for i in range(trainingCoefs.shape[1]):
            dist.append(euclidean_dist(trainingCoefs[:, i].reshape(-1, 1), testingCoefs))
    elif Mode == 2:
        for i in range(trainingCoefs.shape[1]):
            dist.append(manhattan_dist(trainingCoefs[:, i].reshape(-1, 1), testingCoefs))
    elif Mode == 3:
        dist = cosine_dist(trainingCoefs, testingCoefs)
    return np.asarray(dist)

def getIdentificationRank(SimM, rank):
    class_idx = np.argpartition(SimM, rank, axis=0)[:rank, :] // 3
    correctID = 0
    for i in range(SimM.shape[1]):
        if i // 7 in class_idx[:, i]:
            correctID += 1
    IDRate = correctID / SimM.shape[1]
    return IDRate

if __name__ == "__main__":
    facedata = sio.loadmat("facedata.mat")["facedata"]
    trainingfaces, testingfaces = splitData(facedata)
    eigfaces, meanImage = getEigenfaces(trainingfaces)
    trainingCoefs = projectFaces(trainingfaces, eigfaces, meanImage)
    testingCoefs = projectFaces(testingfaces, eigfaces, meanImage)
    checkCluster(testingCoefs)
    IDRates = []
    for Mode in [1, 2, 3]:
        SimM = getSimM(trainingCoefs, testingCoefs, Mode)
        for rank in [1, 2, 3]:
            IDRates.append(getIdentificationRank(SimM, rank))
    IDRates = np.asarray(IDRates).reshape(3, 3)
    print(IDRates)
    plt.show()
