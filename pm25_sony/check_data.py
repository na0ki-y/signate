import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("./data/train.csv", index_col=0) # 学習用データ
test = pd.read_csv("./data/test.csv", index_col=0)   # 評価用データ
sample_submit = pd.read_csv("./data/submit_sample.csv",  index_col=0, header=None) # 応募用サンプルファイル


def plot_check():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111) 
    pg = sns.pairplot(train[['lat','lon','so2_mid',"co_mid",'pm25_mid']])
    plt.xlabel("Time[step]", fontsize=15)
    plt.ylabel("Number of people on the node", fontsize=15)
    plt.grid()
    fig.savefig('./outputs/seaborn_pairplot_default.png')

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
#sns.set_context("paper", 1.5, {"lines.linewidth": 4})
#sns.set_palette("winter_r", 8, 1)
#sns.set('talk', 'whitegrid', 'dark', font_scale=1.5,rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

pg = sns.pairplot(train[['lat','lon','so2_mid',"co_mid",'pm25_mid']])
pg.savefig('./outputs/seaborn_pairplot_default.png')
#plot_check()