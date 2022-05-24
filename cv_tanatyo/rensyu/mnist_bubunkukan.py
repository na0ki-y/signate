
# MNISTをKL展開し、部分空間を図示

import keras
from keras.datasets import mnist
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = np.array(mnist["data"].get("data").value)
label = np.array(mnist["data"].get("label").value)

# 削減次元数
D2 = 30

# 7のデータだけ取る
x = data[:, np.where(label==7)[0]]

# データ次元数
D = 784

# データ数
N = x.shape[1]

# 平均ベクトルを求める
m = np.array([np.average(x[i, :]) for i in range(D)])

# 共分散行列を求める
s = np.zeros([D, D])
for i in range(N):
    s += np.outer(x[:, i] - m, x[:, i] - m)
s = s/N


# 固有値と固有ベクトルを求める
lam, v = np.linalg.eigh(s)

# 固有値の降順に固有ベクトルを並べ替える
v = v[:, np.argsort(lam)[::-1]]
# D2次元の部分空間の基底が得られる
w = v[:, 0:D2]

# np.save("./data/7", v)

for i in range(5):
    plt.subplot(3, 5, i+1).set_aspect('equal')
    plt.title(i+1)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.imshow(v[:,i].reshape(28, 28), interpolation="None", cmap=cm.gray)

for i in range(5):
    plt.subplot(3, 5, i+6).set_aspect('equal')
    plt.title(i+100)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.imshow(v[:,i+99].reshape(28, 28), interpolation="None", cmap=cm.gray)

for i in range(5):
    plt.subplot(3, 5, i+11).set_aspect('equal')
    plt.title(i+780)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.imshow(v[:,i+779].reshape(28, 28), interpolation="None", cmap=cm.gray)

plt.show()