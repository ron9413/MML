import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.sparse.linalg import eigs
import time

def plot_eigenface(eigVec, height, width):
    fig = plt.figure()
    a = fig.add_subplot(4, 3, 2)
    imgplot = plt.imshow(np.reshape(X_mean, (height, width)), cmap='gray')
    for i, vec in enumerate(eigVec.T):
        a = fig.add_subplot(4, 3, i+4)
        fig.tight_layout()
        imgplot = plt.imshow(np.reshape(vec, (height, width)), cmap='gray')

def eigen_analysis(X, num_sample):
    cov_matrix = np.matmul(X, X.T) / (num_sample - 1)
    eigVal, eigVec = eigs(cov_matrix, num_sample-1)
    idx = eigVal.astype(float).argsort()[::-1]
    eigVal = eigVal[idx].astype(float)
    eigVec = eigVec[:, idx].astype(float)
    return eigVal, eigVec

def gram_matrix(X, num_sample):
    cov_matrix = np.matmul(X.T, X) / (num_sample - 1)
    eigVal, eigVec = eig(cov_matrix)
    idx = eigVal.astype(float).argsort()[::-1][:num_sample-1]
    eigVal = eigVal[idx].astype(float)
    eigVec = eigVec[:, idx].astype(float)
    eigVec = np.matmul(X, eigVec)
    eigVec = eigVec/norm(eigVec, axis=0)
    return eigVal, eigVec

def plot_reconstruct(X, V, m):
    fig = plt.figure()
    for i in range(V.shape[1]):
        p = np.matmul(V.T[:i+1], (X - m))
        X_reconstruct = np.matmul(V[:, :i+1], p) + m
        a = fig.add_subplot(3, 3, i+1)
        fig.tight_layout()
        mse = np.mean((X - X_reconstruct) ** 2)
        a.set_title("MSE=" + str(mse))
        img = plt.imshow(np.reshape(X_reconstruct, (height, width)), cmap='gray')

if __name__ == "__main__":
    mat_dict = sio.loadmat("facedata.mat")
    person1 = mat_dict["facedata"][0]
    person1 = np.stack(person1)

    num_sample, height, width = person1.shape
    dim = height * width
    X = person1.reshape(num_sample, dim).T
    X_mean = np.mean(X, axis=1).reshape(dim, 1)
    X = X - X_mean

    start = time.time()
    eigVal, eigVec = eigen_analysis(X, num_sample)
    plot_eigenface(eigVec, height, width)
    end = time.time()
    print(end-start)

    start = time.time()
    eigVal_gram, eigVec_gram = gram_matrix(X, num_sample)
    plot_eigenface(eigVec_gram, height, width)
    end = time.time()
    print(end-start)

    person1_image1 = person1[0].reshape(dim, 1)
    plot_reconstruct(person1_image1, eigVec, X_mean)

    plt.show()
