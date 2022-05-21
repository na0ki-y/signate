import pandas as pd
train = pd.read_csv("./data/train.tsv", sep="\t", index_col=0) # 学習用データ
test = pd.read_csv("./data/test.tsv", sep="\t", index_col=0)   # 評価用データ
sample_submit = pd.read_csv("./data/sample_submit.tsv", sep="\t", index_col=0, header=None) # 応募用サンプルファイル


train = train[["survived","sibsp", "parch", "fare"]]
test = test[["sibsp", "parch", "fare"]]
y = train["survived"] # 目的変数
X = train.drop(["survived"], axis=1) # 目的変数を除いたデータ

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

pred = model.predict_proba(test)[:, 1] 


sample_submit[1] = pred
sample_submit.to_csv('./data/submit.tsv', header=None, sep='\t')