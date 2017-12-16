from my_svm import *
import svm_ref
import numpy as np
import matplotlib.pyplot as pyplot


def test_line_trans():
    data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data2 = np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7]])
    K_line1 = line_trans(data1, data2)
    print(K_line1)
    print()
    K_line2 = np.mat(np.zeros((3, 3)))
    for i in range(3):
        # K_line2[:, i] = svm_ref.kernelTrans(data1, data2[i, :], ('lin', 1))
        K_line2[:, i] = svm_ref.kernelTrans(np.mat(data1), np.mat(data2[i, :]), ('lin', 1))
    print(K_line2)


def test_rbf_trans():
    data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], ])
    data2 = np.array([[2, 4, 6], [3, 5, 7]])
    gamma = 3
    K1 = rbf_trans(data1, data2, gamma)
    print(K1)
    print()
    K2 = np.mat(np.zeros((4, 2)))
    for i in range(2):
        K2[:, i] = svm_ref.kernelTrans(np.mat(data1), np.mat(data2[i, :]), ('rbf', np.sqrt(1/gamma)))
    print(K2)


def test_meta():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5],[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5]])
    m = len(data)
    y = np.array([1, -1, 1, -1, 1, -1, 1, -1]).reshape((-1, 1))
    C = 2
    toler = 0.0001
    gamma = 3
    k_tup1 = ('rbf', gamma)
    k_tup2 = ('rbf', np.sqrt(1/gamma))
    meta = Meta(data, y, C, toler, k_tup1)
    oS = svm_ref.optStruct(np.mat(data), np.mat(y), C, toler, k_tup2)
    for i in range(oS.m):
        oS.eCache[i, 0] = 1
    # alpha = np.random.uniform(0, C, (len(data), 1))
    alpha = np.array([[0.16982751],
                      [0.77701053],
                      [1.70981247],
                      [1.84589539],
                      [0.16982751],
                      [0.77701053],
                      [1.70981247],
                      [1.84589539]
                      ])
    meta.a = alpha.reshape((-1, 1))
    oS.alphas = np.mat(alpha.reshape((-1, 1)))
    meta.calc_e()
    # print(meta.K)

    print(meta.e_cache)
    print(svm_ref.calcE(oS))

    # K2 = np.mat(np.zeros((4, 4)))
    # for i in range(4):
    #     K2[:, i] = svm_ref.kernelTrans(np.mat(data), np.mat(data[i, :]), ('rbf', np.sqrt(1 / gamma)))
    # print(K2)
    # print(K2 - meta.K)
    # print()

    print(meta.a)
    print(oS.alphas)
    print('alpha above')
    print()

    print(meta.e_cache)
    for k in range(m):
        print(k)
        # print(meta.e_cache)
        inner_l(k, meta)
        # j, Ej = select_j(i, meta)
        # print(j, Ej)
    # for k in range(m):
    #     print(k)
    #     # print(svm_ref.calcE(oS))
    #     # inner_l(k, meta)
    #     svm_ref.innerL(k, oS)
    #     print()
    # print()


def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat).reshape((-1, 1))


def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels)
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        # print(np.shape(w))
        # print(np.shape(X[i,:].T))
        # print(np.shape(alphas[i]))
        # print(np.shape(labelMat[i]))
        # print(np.shape(alphas[i]*labelMat[i]))
        # print(np.shape(np.multiply(alphas[i]*labelMat[i], X[i,:].T)))
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


def test_smo():
    print('test smo')
    filename = './practice/testSet.txt'
    dataMat, labelMat = loadDataSet(filename)
    C = 2
    toler = 1e-6
    max_iter = 100
    k_tup = ('rbf', 1)
    # meta = sim_smo(dataMat, labelMat, C, toler, max_iter, k_tup)
    meta = smo(dataMat, labelMat, C, toler, max_iter, k_tup)
    result = predict(meta, dataMat)
    # check1 = (result > 0) * (labelMat > 0)
    # check2 = (result < 0) * (labelMat < 0)
    # check = check1 + check2
    check = (result == labelMat)
    print(result)
    print('sum, check', np.sum(check))
    # check = (result == labelMat)
    # print(check)
    # print(np.sum(check))
    # print(len(check))
    # w = calcWs(np.mat(meta.a), dataMat, labelMat)
    # print(w)
    # print(meta.b)
    # (-1 * w[0] - meta.b) /
    # print(dataMat)
    # print(labelMat)


def test_smo2():
    filename = './practice/testSet.txt'
    C = 200
    toler = 0
    max_iter = 100
    k_tup = ('lin', 1)
    dataMat, labelMat = svm_ref.loadDataSet(filename)
    b, alphas = svm_ref.smoP(dataMat, labelMat, C, toler, max_iter)
    w = svm_ref.calcWs(alphas, dataMat, labelMat)
    print(w)
    print(b)


if __name__ == '__main__':
    test_smo()
    # test_smo()
    # test_rbf_trans()
