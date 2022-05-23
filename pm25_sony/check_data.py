from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
train = pd.read_csv("./data/train.csv", index_col=0) # 学習用データ
test = pd.read_csv("./data/test.csv", index_col=0)   # 評価用データ
sample_submit = pd.read_csv("./data/submit_sample.csv",  index_col=0, header=None) # 応募用サンプルファイル

day_cal=[]
kizyun_day=datetime.date(2019,1,1)
for index, row in train.iterrows():
    day_cal.append((datetime.date(row["year"],row["month"],row["day"])-kizyun_day).days)
train=train.assign(day_cal=day_cal)

day_cal=[]
kizyun_day=datetime.date(2019,1,1)
for index, row in test.iterrows():
    day_cal.append((datetime.date(row["year"],row["month"],row["day"])-kizyun_day).days)
test=test.assign(day_cal=day_cal)

country_dict={}
country_list=[]
for index, row in train.iterrows():
    country_dict.setdefault(row["Country"], len(country_dict))
    country_list.append(country_dict[row['Country']])
train=train.assign(country_list=country_list)
print(country_dict)

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

print(train)

pg = sns.pairplot(train[['lat','lon',"country_list",'pm25_mid']])
pg.savefig('./outputs/seaborn_pairplot_default_country.png')


pg = sns.pairplot(train[['lat','lon','pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_latlon.png')

pg = sns.pairplot(train[["co_var","co_max","co_min",'pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_co.png')

pg = sns.pairplot(train[['year','month','day',"day_cal",'pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_day.png')

pg = sns.pairplot(train[['co_mid','o3_mid','so2_mid','no2_mid','pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_mid.png')

pg = sns.pairplot(train[['co_var','o3_var','so2_var','no2_var','pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_var.png')

pg = sns.pairplot(train[['ws_var','ws_min','ws_max','pm25_mid']])
#pg.savefig('./outputs/seaborn_pairplot_default_ws.png')
#plot_check()