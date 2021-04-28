import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
clf = linear_model.LinearRegression()

gdp = pd.read_csv('import/GDP_JAPAN.csv',
                  parse_dates=['DATE'], index_col='DATE')
iip = pd.read_csv('import/IIP.csv', parse_dates=['DATE'], index_col='DATE')


gdp.index = pd.to_datetime(gdp.index, format='%m/%d/%Y').strftime('%Y-%m-01')
iip.index = pd.to_datetime(iip.index, format='%m/%d/%Y').strftime('%Y-%m-01')

df = pd.merge(gdp, iip, left_index=True, right_index=True, how='outer')

# 日本の会計年度が4月始まりのため、4月を起点としている。
df['period'] = pd.to_datetime(df.index.to_series()).apply(
    lambda x: 3 if x.month in [1, 4, 7, 10] else (1 if x.month in [2, 5, 8, 11] else 2))

df = df[df.index != '2013-01-01']

df2 = pd.DataFrame(columns=[
                   'GDP_CYOY', 'IIP_YOY_Q1', 'IIP_YOY_Q2', 'IIP_YOY_Q3'])
for date, GDP_CYOY, IIP_YOY, period in zip(df.index, df.GDP_CYOY, df.IIP_YOY, df.period):

    if period == 1:
        q1 = IIP_YOY
    elif period == 2:
        q2 = IIP_YOY
    else:
        record = pd.Series([GDP_CYOY, q1, q2, IIP_YOY],
                           index=df2.columns, name=date)
        df2 = df2.append(record)

df2.index.name = 'DATE'


# 目的変数(Y)、説明変数(X)
Y = np.array(df2['GDP_CYOY'])
X = np.array(df2[['IIP_YOY_Q1', 'IIP_YOY_Q2', 'IIP_YOY_Q3']])

# 予測モデルを作成
clf.fit(X, Y)

# 偏回帰係数
ab = pd.DataFrame({"Name": ['IIP_YOY_Q1', 'IIP_YOY_Q2', 'IIP_YOY_Q3'],
                  "Coefficients": clf.coef_}).sort_values(by='Coefficients')
for index, row in ab.iterrows():
    if row.Name == 'IIP_YOY_Q3':
        b3 = row.Coefficients
    elif row.Name == 'IIP_YOY_Q2':
        b2 = row.Coefficients
    else:
        b1 = row.Coefficients
# 切片
print(clf.intercept_)
a = clf.intercept_

df2['NOWCAST'] = df2.apply(lambda x: a + b1*x['IIP_YOY_Q1'] +
                           b2*x['IIP_YOY_Q2'] + b3*x['IIP_YOY_Q3'], axis=1)

newdf2 = df2.copy()
newdf2 = newdf2.drop('IIP_YOY_Q1', axis=1)
newdf2 = newdf2.drop('IIP_YOY_Q2', axis=1)
newdf2 = newdf2.drop('IIP_YOY_Q3', axis=1)

if __name__ == '__main__':
    print("ok")
    #pd.set_option('display.max_columns', 4)
    # print(df)
    plt.figure()
    newdf2.plot()
    plt.savefig('export/nowcaste.png')
    plt.close('all')
