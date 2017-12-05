import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svmutil import *
import numpy as np
import sys

kernel = sys.argv[1]

def loadData():
    mat_dict = sio.loadmat("usps.mat")
    X = mat_dict['X']
    Y = mat_dict['Y']
    label_train = mat_dict['label_train'].flatten()
    label_test = mat_dict['label_test'].flatten()
    cvfold = mat_dict['cvfold'].flatten() - 1
    return X, Y, label_train, label_test, cvfold

def ovrtrain(y, x, cmd):
    labelSet = np.unique(y)
    labelSetSize = len(labelSet)
    models = []
    for i in range(labelSetSize):
        model = svm_train((y==labelSet[i]).astype(float).tolist(), x.tolist(), cmd)
        models.append(model)
    return {'models': models, 'labelSet': labelSet}

def ovrpredict(y, x, model):
    labelSet = model['labelSet']
    labelSetSize = len(labelSet)
    models = model['models']
    decv = np.zeros((len(y), labelSetSize))
    for i in range(labelSetSize):
        l, a, d = svm_predict((y==labelSet[i]).astype(float).tolist(), x.tolist(), models[i])
        decv[:, i] = np.asarray(d).flatten() * (2 * models[i].get_labels()[0] - 1)
    pred = np.argmax(decv, axis=1)
    pred = labelSet[pred]
    ac = np.sum(y==pred) / len(x)
    return pred, ac, decv

def get_cv_ac(y, x, param, cvfold):
    length = len(y)
    ac = 0
    for i in range(len(cvfold)):
        print('cvfold{}'.format(i))
        trainIdx, testIdx = get_cv_idx(length, cvfold[i])
        model = ovrtrain(y[trainIdx], x[:, trainIdx].T, param)
        pred, a, decv = ovrpredict(y[testIdx], x[:, testIdx].T, model)
        ac = ac + np.sum(y[testIdx]==pred)
    ac = ac / length
    return ac

def my_svm_predict(y, x, model):
    pass

def get_cv_idx(length, cvfold_i):
    testIdx = cvfold_i.flatten()
    trainIdx = np.ones(length, dtype=bool)
    trainIdx[testIdx] = False
    return trainIdx, testIdx

def get_best_C(label_train, X, cvfold, kernel):
    cv_list = []
    bestcv = 0
    bestC = 0
    bestSigma = 0
    if kernel == 'linear':
        for C in C_list:
            cmd = '-t 0 -c {}'.format(C)
            print(cmd)
            cv = get_cv_ac(label_train, X, cmd, cvfold)
            cv_list.append(cv)
            if cv >= bestcv:
                bestcv = cv
                bestC = C
    else:
        for C in C_list:
            tmp = []
            for s in sigma:
                gamma = 1. / (2 * s**2)
                cmd = '-t 2 -c {} -g {}'.format(C, gamma)
                print(cmd)
                cv = get_cv_ac(label_train, X, cmd, cvfold)
                tmp.append(cv)
                if cv >= bestcv:
                    bestcv = cv
                    bestC = C
                    bestSigma = s
            cv_list.append(tmp)
    return bestC, bestSigma, cv_list

def plot_cv_recognition_rate(cv_list, kernel):
    if kernel == 'linear':
        plt.plot(C_list, cv_list)
        plt.xscale('log')
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xticks(np.log10(sigma))
        ax.set_xticklabels(sigma)
        ax.set_yticks(np.log10(C_list))
        ax.set_yticklabels(C_list)
        x, y = np.meshgrid(sigma, C_list)
        x = np.log10(x)
        y = np.log10(y)
        z = cv_list
        ax.plot_surface(x, y, z)


def svm_clf(X, Y, label_train, label_test, cvfold, kernel):
    bestC, bestSigma, cv_list = get_best_C(label_train, X, cvfold, kernel)
    '''
    100 10
    [[0.16376354409546015, 0.16376354409546015, 0.51296118502263066, 0.83829378686051292, 0.75970374434233989],
     [0.16376354409546015, 0.16376354409546015, 0.28816348923330132, 0.85392950212590868, 0.79728432313811548],
     [0.16376354409546015, 0.16376354409546015, 0.28761486764504185, 0.93896584830613083, 0.79399259360855845],
     [0.16376354409546015, 0.16376354409546015, 0.28308873954190095, 0.97174598820463587, 0.86161020436154168],
     [0.16376354409546015, 0.16376354409546015, 0.28020847620353861, 0.98093539980798239, 0.92717048415855163],
     [0.16376354409546015, 0.16376354409546015, 0.28020847620353861, 0.98093539980798239, 0.94541215196817996],
     [0.16376354409546015, 0.16376354409546015, 0.28020847620353861, 0.97956384583733369, 0.9510355232478398]]
    '''
    print(bestC, bestSigma, cv_list)
    plot_cv_recognition_rate(cv_list, kernel)
    if kernel == 'linear':
        cmd = '-t 0 -c {}'.format(bestC)
    else:
        bestGamma = 1. / (2 * bestSigma**2)
        cmd = '-t 2 -c {} -g {}'.format(bestC, bestGamma)
    print(cmd)
    model = ovrtrain(label_train, X.T, cmd)
    pred, ac, decv = ovrpredict(label_test, Y.T, model)
    print(ac)
    ### linear: ac = 0.914299950174
    ### nonlinear: ac = 0.952167414051
    plt.show()

if __name__ == "__main__":
    C_list = [10 ** e for e in range(-3, 4)]
    sigma = [10 ** e for e in range(-2, 3)]
    X, Y, label_train, label_test, cvfold = loadData()
    svm_clf(X, Y, label_train, label_test, cvfold, kernel)
