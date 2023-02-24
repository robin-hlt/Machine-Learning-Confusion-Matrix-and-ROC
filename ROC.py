import numpy as np
import copy
import matplotlib.pyplot as plt
import random


def confusion_matrix(y_true, y_pred, n):
    matrix = np.zeros((n, n))
    if n == 2:  # 二分类
        for i in range(len(y_true)):
            row = int(y_true[i]-1)
            col = int(y_pred[i]-1)
            matrix[row, col] += 1
    else:  # 多分类
        for i in range(len(y_true)):
            row = int(y_true[i])
            col = int(y_pred[i])
            matrix[row, col] += 1
    return matrix


def roc(y_true, y_pred):
    sort_pred = copy.deepcopy(y_pred)
    sort_pred.sort(reverse=True)
    TPR = []
    FPR = []
    for i in range(len(sort_pred)):
        threshold = sort_pred[i]
        label_pred = np.array(copy.deepcopy(y_pred))
        label_true = np.array(copy.deepcopy(y_true))
        label_pred[label_pred > threshold] = 1
        label_pred[label_pred <= threshold] = 0
        label_true[label_true != 1] = 0
        matrix = confusion_matrix(list(label_true), list(label_pred), 2)
        tpr = matrix[0][0]/(matrix[0][0]+matrix[0][1])
        fpr = matrix[1][0]/(matrix[1][0]+matrix[1][1])
        TPR.append(tpr)
        FPR.append(fpr)
    x = np.array(FPR)
    y = np.array(TPR)
    plt.plot(x, y)
    plt.show()


random.seed(0)
y_pred = [0.5+random.uniform(-0.2, 0.2) for _ in range(100)]
y_true = [random.randint(0, 3) for _ in range(100)]
roc(y_true, y_pred)

