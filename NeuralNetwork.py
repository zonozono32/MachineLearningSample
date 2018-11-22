import random
import numpy as np
from math import exp
import matplotlib.pyplot as plt

INPUT = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
TEACH = np.array([[0], [1], [1], [0]])

_, IN_NUM = np.shape(INPUT)
_, OUT_NUM = np.shape(TEACH)
HIDDEN_NUM = 3
PAT_NUM = 4
ETA = 0.2

w1 = np.random.rand(IN_NUM + 1, HIDDEN_NUM) - 0.5
w2 = np.random.rand(HIDDEN_NUM + 1, OUT_NUM) - 0.5

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def output(X):
    yy1 = np.zeros(HIDDEN_NUM)
    for j in range(HIDDEN_NUM):
        yy1[j] = sum(w1[:IN_NUM, j] * X)

    # バイアス加算
    yy1 += w1[IN_NUM]
    yy1 = sigmoid(yy1)

    out = np.zeros(OUT_NUM)
    for j in range(OUT_NUM):
        out[j] = sum(w2[:HIDDEN_NUM, j] * yy1)

    # バイアス加算
    out += w2[HIDDEN_NUM]
    out = sigmoid(out)

    return yy1, out

def learn(X, y):
    yy1, out = output(X)
    error = 0

    sub = y - out
    error = sum(sub * sub)

    delta = out * (1.0 - out) * (y - out)

    # 出力層の学習
    w2[:HIDDEN_NUM] += ETA * delta * np.reshape(yy1, (HIDDEN_NUM, 1))
    w2[HIDDEN_NUM] += ETA * delta

    # 隠れ層の学習
    for j in range(HIDDEN_NUM):
        delta2 = sum(delta * w2[j])

        for i in range(IN_NUM):
            w1[i][j] += ETA * yy1[j] * (1.0 - yy1[j]) * delta2 * X[i]

        w1[i][j] += ETA * yy1[j] * (1.0 - yy1[j]) * delta2

    return error

error_array = []

for i in range(30000):
    index = int(random.random() * PAT_NUM)
    e = learn(INPUT[index], TEACH[index])
    print(i, "回目：error = ", str(e))
    error_array.append(e)

for X_test in INPUT:
    _, result = output(X_test)
    print(X_test, "->", result)

plt.plot(range(30000), error_array)
plt.show()
