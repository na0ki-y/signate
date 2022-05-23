import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
import numpy as np
import os
# LabelEncoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold

from sklearn.metrics import mean_squared_error
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix



import json
import datetime
OBJPRM="charges"
THIS_PROG_NAME="first.py"
INPUT_TRAIN="../data/train.csv"
INPUT_TEST="../data/test.csv"
SAMPLE_SUB="../data/sample_submit.csv"


def make_XY(train,test):
    train_Y = train[OBJPRM] # 目的変数
    train_X = train.drop([OBJPRM], axis=1) # 目的変数を除いたデータ
    test_X=test
    return train_X,train_Y,test_X

def plot_XY():
    pass

def labelencodeing(train,test):
    #trainとtestを結合
    data=pd.concat([train,test])
    #labelencoding
    feat=data.columns
    print(data)
    
    catfeats= list(data.select_dtypes(include= "object").columns) #objectのカラムをリストに入れる
    print(catfeats)
    le= LabelEncoder() #ラベルエンコーダーをインスタンス化して使えるようにする
    labels={}
    for feat in catfeats: 
        le.fit(data[feat].astype(str))
        label={feat:list(le.classes_)}
        labels.update(label)
        data[feat]=le.transform(data[feat].astype(str))
    #trainとtestを分割
    train=data[data[OBJPRM].notna()]#欠損しているか　目的変数が
    test=data[data[OBJPRM].isna()]
    test=test.drop([OBJPRM], axis=1)#目的変数が含まれてしまうので除く
    return train,test,catfeats

def pred_logi(train_X,train_Y,test_X):
    model = LogisticRegression()
    model.fit(train_X, train_Y)

    pred = model.predict_proba(test_X)[:, 1] 
    train_rmse=np.sqrt(mean_squared_error(train_Y, model.predict_proba(train_X)[:,1]))
    return pred,train_rmse,model

def pred_svm(train_X,train_Y,test_X): 
    #LinearSVMのインスタンスを作成
    Linsvc = svm.LinearSVC(random_state=0, max_iter=3000)
    
    #LinearSVMでデータを学習
    Linsvc.fit(train_X,train_Y)
    pred = Linsvc.predict(test_X)
    train_results={}
    train_results["co_matrix"]=str(confusion_matrix(y_true = train_Y, y_pred = Linsvc.predict(train_X)))
    train_results["acc"]=Linsvc.score(train_X,train_Y)

    return pred,train_results,Linsvc

def pred_lightgpm(train_X,train_Y,test_X):
#https://rightcode.co.jp/blog/information-technology/lightgbm-useful-for-kaggler
    # LightGBM用のデータセットに入れる
    #import lightgbm as lgb

    lgb_train = lgb.Dataset(train_X, train_Y)
    #lgb.test = lgb.Dataset(x_test, y_test)
    # 評価基準を設定する 
    params = {'metric' : 'rmse'}
    gbm = lgb.train(params, lgb_train)
    # テストデータを用いて予測精度を確認する
    test_predicted = gbm.predict(test_X)
    return test_predicted

def first_read_plot_pred(dropparms,out_path):
    out_json={}
    out_json["date"] = str(datetime.datetime.now())
    out_json["prog_name"]=THIS_PROG_NAME
    print("START")
    train = pd.read_csv(INPUT_TRAIN , index_col=0) # 学習用データ
    test = pd.read_csv( INPUT_TEST, index_col=0)   # 評価用データ
    sample_submit = pd.read_csv(SAMPLE_SUB,  index_col=0, header=None) # 応募用サンプルファイル

    
    out_json["INPUT_TRAIN"]=INPUT_TRAIN
    out_json["INPUT_TEST"]=INPUT_TEST
    out_json["SAMPLE_SUN"]=SAMPLE_SUB

    if out_path=="../result/first_SVM_labelencorde/":
        train,test,catfeats=labelencodeing(train,test)
        out_json["catfeats"]=catfeats


    train_X,train_Y,test_X=make_XY(train,test)
    out_json["OBJPRM"]=OBJPRM
    
    if out_path=="../result/first_Logi_nolabelencorde/" or out_path=="../result/first_SVM_nolabelencorde/":
        train_X=train_X.drop(dropparms, axis=1) # 目的変数を除いたデータ
        test_X=test_X.drop(dropparms, axis=1) # 目的変数を除いたデータ
        out_json["dropoarms"]=dropparms


    if out_path=="../result/first_Logi_nolabelencorde/":
        pred,train_rmse,model=pred_logi(train_X,train_Y,test_X)
        out_json["model_name"]="logosticReg"
        out_json["train_rmse"]=train_rmse
    elif out_path=="../result/first_SVM_nolabelencorde/" or out_path=="../result/first_SVM_labelencorde/":
        pred,train_results,model=pred_svm(train_X,train_Y,test_X)
        out_json["model_name"]="SVM"
        out_json["train_results"]=train_results
    
    out_json["out_path"]=out_path
    if not os.path.isdir(out_path):
        try:
            os.makedirs(out_path)
        except FileExistsError as e:
            print(e)

    sample_submit[1] = pred
    sample_submit.to_csv(out_path+'submit_first.csv', header=None)
    with open(out_path+'output.json', 'w') as f:
        json.dump(out_json, f,ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    with open(out_path+'model.sav', 'wb') as f:
        pickle.dump(model, f)
    print("end")

    
if __name__=="__main__":
    dropparms=["sex","smoker","region"]
    out_path="../result/first_Logi_nolabelencorde/"
    first_read_plot_pred(dropparms,out_path)
    print("../result/first_Logi_nolabelencorde/")

    dropparms=["sex","smoker","region"]
    out_path="../result/first_SVM_nolabelencorde/"
    first_read_plot_pred(dropparms,out_path)
    print("../result/first_SVM_nolabelencorde/")

    dropparms=None
    out_path="../result/first_SVM_labelencorde/"
    first_read_plot_pred(dropparms,out_path)
