#https://intellectual-curiosity.tokyo/2019/07/02/%E3%82%AA%E3%83%AA%E3%82%B8%E3%83%8A%E3%83%AB%E3%81%AE%E7%94%BB%E5%83%8F%E3%81%8B%E3%82%89%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B%E6%96%B9/
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from matplotlib import cm
DATADIR = "../train"
CATEGORIES = ["000", "001"]
DIM=(4032, 3024, 3)
def print_one():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, image_name), )
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            plt.imshow(img_array, cmap='gray')
            plt.savefig("./outputs/one_pic.pdf")
            
            break
        break
  
    print(img_array.shape)#(4032, 3024, 3)
    print(img_array)
IMG_SIZE = 50
training_data = []

def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass
    
        random.shuffle(training_data)  # データをシャッフル
        X_train = []  # 画像データ
        y_train = []  # ラベル情報
        # データセット作成
        for feature, label in training_data:
            X_train.append(feature)
            y_train.append(label)
        # numpy配列に変換
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # データセットの確認
        for i in range(0, 4):
            print("学習データのラベル：", y_train[i])
            plt.subplot(2, 2, i+1)
            plt.axis('off')
            plt.title(y_train[i])
            plt.imshow(X_train[i], cmap='gray')
        plt.savefig("./outputs/train_{}.pdf".format(category))
    

def koyuti():
    D=4032*3024*3
    W=[]
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        training_data=[]
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)  # 画像読み込み
                img_array=cv2.resize(img_array, (50,50))  # 画像のリサイズ
                img_resize_array = np.ravel(img_array)
                training_data.append(img_resize_array)  # 画像データ
            except Exception as e:
                pass
            # 平均ベクトルを求める
        training_data=np.array(training_data).T
        D,N=np.array(training_data).shape
        m = np.array([np.average(training_data[i,:]) for i in range(D)])#D*N

        # 共分散行列を求める
        s = np.zeros([D, D])
        for i in range(N):
            s += np.outer(training_data[:,i] - m, training_data[:,i] - m)#D*N
        s = s/N
        # 固有値と固有ベクトルを求める
        lam, v = np.linalg.eigh(s)

        # 固有値の降順に固有ベクトルを並べ替える
        v = v[:, np.argsort(lam)[::-1]]
        # D2次元の部分空間の基底が得られる
        D2=30
        W.append(v[:, 0:D2] )#D*D2

        # np.save("./data/7", v)

        for i in range(5):
            plt.subplot(3, 5, i+1).set_aspect('equal')
            plt.title(i+1)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.imshow(v[:,i].reshape(50, 50), interpolation="None", cmap=cm.gray)

        for i in range(5):
            plt.subplot(3, 5, i+6).set_aspect('equal')
            plt.title(i+100)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.imshow(v[:,i+99].reshape(50, 50), interpolation="None", cmap=cm.gray)

        for i in range(5):
            plt.subplot(3, 5, i+11).set_aspect('equal')
            plt.title(i+780)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.imshow(v[:,i+779].reshape(50, 50), interpolation="None", cmap=cm.gray)

        plt.savefig("./outputs/koyubekutoru_{}.pdf".format(category))

        print(type(np.array(W[0])))#D*D2
    return W
    

def bubunkukanhou():
    W=koyuti()
    k_num=2
    CONF=np.darray(k_num*k_num).reshape(k_num,k_num)
    for i in range(N):
        len_list=[]
        for i in range(k_num):
            len_list.append(W[0]*training_data[:,i])#nrom
        maxindex = np.argmax(len_list)#indexが予測

if __name__=="__main__":
    print_one()
    create_training_data()
    koyuti()
