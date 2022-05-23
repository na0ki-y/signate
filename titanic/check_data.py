import pandas as pd
import seaborn as sns
train = pd.read_csv("./data/train.tsv", sep="\t", index_col=0) # 学習用データ
test = pd.read_csv("./data/test.tsv", sep="\t", index_col=0)   # 評価用データ
sample_submit = pd.read_csv("./data/sample_submit.tsv", sep="\t", index_col=0, header=None) # 応募用サンプルファイル
pg = sns.pairplot(train)
pg.savefig('./seaborn_pairplot_default.png')