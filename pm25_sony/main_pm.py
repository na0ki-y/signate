'''
20220520->23
https://stats.biopapyrus.jp/sparse-modeling/python-glmnet.html
https://signate.jp/competitions/624/data
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,RidgeCV
import datetime
train = pd.read_csv("./data/train.csv", index_col=0) # 学習用データ
test = pd.read_csv("./data/test.csv", index_col=0)   # 評価用データ
sample_submit = pd.read_csv("./data/submit_sample.csv",  index_col=0, header=None) # 応募用サンプルファイル

country_dict={}
country_list=[]
for index, row in train.iterrows():
    country_dict.setdefault(row["Country"], len(country_dict))
    country_list.append(country_dict[row['Country']])
train=train.assign(country_list=country_list)


country_list=[]
for index, row in test.iterrows():
    country_list.append(country_dict[row['Country']])
test=test.assign(country_list=country_list)


train_Y = train["pm25_mid"] # 目的変数
train_X = train.drop(["pm25_mid","Country","City"], axis=1) # 目的変数を除いたデータ
test_X = test.drop(["Country","City"], axis=1) # 目的変数を除いたデータ

day_cal=[]
kizyun_day=datetime.date(2019,1,1)
for index, row in train.iterrows():
    day_cal.append((datetime.date(row["year"],row["month"],row["day"])-kizyun_day).days)
#train=train.assign(day_cal=day_cal)

day_cal=[]
kizyun_day=datetime.date(2019,1,1)
for index, row in test.iterrows():
    day_cal.append((datetime.date(row["year"],row["month"],row["day"])-kizyun_day).days)
#test=test.assign(day_cal=day_cal)

train_X = train_X.drop(["year","month","day"], axis=1) # 目的変数を除いたデータ
test_X = test_X.drop(["year","month","day"], axis=1) # 目的変数を除いたデータ


########CV


scaler = StandardScaler()
clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)

scaler.fit(train_X)
clf.fit(scaler.transform(train_X), train_Y)
print(clf.alpha_)

print(clf.coef_)

print(clf.intercept_)

y_pred = clf.predict(scaler.transform(test_X))

sample_submit[1] = y_pred
sample_submit.to_csv('./data/submit.csv', header=None)

print(y_pred)