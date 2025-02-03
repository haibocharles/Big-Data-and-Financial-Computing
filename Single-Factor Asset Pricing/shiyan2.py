
#单因子
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import f
import warnings
plt.figure(1)
#选择708股票
onedata=pd.read_csv('onedata.csv')
onedata.columns= ['code','date','close','rf']
onedata.dropna(inplace=True)
stk_onecodes = np.unique(onedata['code'].values)
print(stk_onecodes)
stock_data1 = onedata[onedata['code'] == stk_onecodes[0]]#重要
stock_data1['return'] = np.log(stock_data1['close']) - np.log(stock_data1['close'].shift(periods=1))#对数收益#重要
stock_data1.dropna(inplace=True)
ind = (stock_data1['return'] >= -0.1) & (stock_data1['return'] <= 0.1)
stock_data1 = stock_data1.loc[ind, :]
stock_data1['date']=pd.to_datetime(stock_data1['date'])
print(stock_data1)
plt.subplot(2,2,1)
stock_data1['return'].plot()#收益率画图
# plt.show()
data300=pd.read_csv('300dailydata.csv',encoding='GB2312')
data300.columns=['stkcode','date','close']
data300['date']=pd.to_datetime(data300['date'])
data300.dropna(inplace=True)
data300['return'] = np.log(data300['close']) - np.log(data300['close'].shift(periods=1))#对数收益率
merge_data = pd.merge(left=stock_data1[['date', 'return', 'rf']],right=data300[['date','return']],
                      on='date',
                      how='inner')#第一个为股票第二为指数按照日期去拼
merge_data.columns = ['date', 'return_708', 'rfreturn', 'return_指数']
merge_data.dropna(inplace=True)
print(merge_data)
#进行708单因子检验
stk_ret = merge_data['return_708'].values
rf_ret = merge_data['rfreturn'].values
ind_ret = merge_data['return_指数'].values
plt.subplot(2,2,2)
plt.plot(ind_ret - rf_ret, stk_ret - rf_ret, 'o', ms=5, mfc='w', lw=2)#股票收益减去无风险
plt.xlabel(r'$r_m - r_f$', fontsize = 20)
plt.ylabel(r'$r_i - r_f$', fontsize = 20)
#回归利用values
x = sm.add_constant(ind_ret-rf_ret)#定义x
y = stk_ret - rf_ret
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())#CAPM模型做708时间序列检验
# plt.show()
#选择20股票
data20=pd.read_excel('20data.xls')
data20.columns=['stkcode','date','close','rf']
stk_onecodes=np.unique(data20['stkcode'].values)
stock_data2 = data20[data20['stkcode'] == stk_onecodes[0]]
stock_data2['return'] = np.log(stock_data2['close']) - np.log(stock_data2['close'].shift(periods=1))#对数收益
stock_data2.dropna(inplace=True)
ind = (stock_data2['return'] >= -0.1) & (stock_data2['return'] <= 0.1)
# 第二行使用 `.loc` 选择仅包含 `ind` 为 True 和所有列（表示为 `:`）的行。这将导致一个新的 DataFrame，其中仅包含 'return' 列值介于 -0.1 和 0.1（含）之间的行。
stock_data2 = stock_data2.loc[ind, :]
plt.subplot(2,2,3)
stock_data2['return'].plot()
merge_data = pd.merge(left=merge_data,right=stock_data2[['date','return']],
                      on='date',
                      how='inner')#第一个为股票第二为指数按照日期去拼
merge_data.columns = ['date', 'return_708', 'rfreturn', 'return_指数','return_20']
merge_data.dropna(inplace=True)
stk_ret = merge_data['return_20'].values
rf_ret = merge_data['rfreturn'].values
ind_ret = merge_data['return_指数'].values
plt.subplot(2,2,4)
#20进行单子产检验
plt.plot(ind_ret - rf_ret, stk_ret - rf_ret, 'o', ms=5, mfc='w', lw=2)#股票收益减去无风险
plt.xlabel(r'$r_m - r_f$', fontsize = 20)
plt.ylabel(r'$r_i - r_f$', fontsize = 20)
x = sm.add_constant(ind_ret-rf_ret)#定义x
y = stk_ret - rf_ret
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())#20进行单子产检验
plt.show()
#输入574
plt.figure(2)
data547=pd.read_excel('547data.xls')
data547.columns=['stkcode','date','close','rf']
stk_onecodes=np.unique(data547['stkcode'].values)
stock_data3 = data547[data547['stkcode'] == stk_onecodes[0]]
stock_data3['return'] = np.log(stock_data3['close']) - np.log(stock_data3['close'].shift(periods=1))#对数收益
stock_data3.dropna(inplace=True)
ind = (stock_data3['return'] >= -0.1) & (stock_data3['return'] <= 0.1)
stock_data3 = stock_data3.loc[ind, :]
plt.subplot(2,2,1)
stock_data3['return'].plot()
merge_data = pd.merge(left=merge_data,right=stock_data3[['date','return']],
                      on='date',
                      how='inner')#第一个为股票第二为指数按照日期去拼
merge_data.columns = ['date', 'return_708', 'rfreturn', 'return_指数','return_20','return_547']
merge_data.dropna(inplace=True)
stk_ret = merge_data['return_547'].values
rf_ret = merge_data['rfreturn'].values
ind_ret = merge_data['return_指数'].values
plt.subplot(2,2,2)
plt.plot(ind_ret - rf_ret, stk_ret - rf_ret, 'o', ms=5, mfc='w', lw=2)#股票收益减去无风险
plt.xlabel(r'$r_m - r_f$', fontsize = 20)
plt.ylabel(r'$r_i - r_f$', fontsize = 20)
#20进行单资产检验
x = sm.add_constant(ind_ret-rf_ret)#定义x
y = stk_ret - rf_ret
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
# plt.show()
# 输入982
data982=pd.read_excel('982data.xls')
data982.columns=['stkcode','date','close','rf']
stk_onecodes=np.unique(data982['stkcode'].values)
stock_data4 =data982[data982['stkcode'] == stk_onecodes[0]]
stock_data4['return'] = np.log(stock_data4['close']) - np.log(stock_data4['close'].shift(periods=1))#对数收益
stock_data4.dropna(inplace=True)
ind = (stock_data4['return'] >= -0.1) & (stock_data4['return'] <= 0.1)
stock_data4 = stock_data4.loc[ind, :]
plt.subplot(2,2,3)
#982收益率画图
stock_data4['return'].plot()
merge_data = pd.merge(left=merge_data,right=stock_data4[['date','return']],
                      on='date',
                      how='inner')#第一个为股票第二为指数按照日期去拼
merge_data.columns = ['date', 'return_708', 'rfreturn', 'return_指数','return_20','return_547','return_982']
merge_data.dropna(inplace=True)
stk_ret = merge_data['return_982'].values
rf_ret = merge_data['rfreturn'].values
ind_ret = merge_data['return_指数'].values
plt.subplot(2,2,4)
#对982进行单资产检验
plt.plot(ind_ret - rf_ret, stk_ret - rf_ret, 'o', ms=5, mfc='w', lw=2)#股票收益减去无风险
plt.xlabel(r'$r_m - r_f$', fontsize = 20)
plt.ylabel(r'$r_i - r_f$', fontsize = 20)
x = sm.add_constant(ind_ret-rf_ret)#定义x
y = stk_ret - rf_ret
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
plt.show()
print(merge_data)
#输入2347
# plt.figure(3)
# data2347=pd.read_excel('2347data.xls')
# data2347.columns=['stkcode','date','close','rf']
# stk_onecodes=np.unique(data2347['stkcode'].values)
# stock_data5 =data2347[data2347['stkcode'] == stk_onecodes[0]]
# stock_data5['return'] = np.log(stock_data5['close']) - np.log(stock_data5['close'].shift(periods=1))#对数收益
# stock_data5.dropna(inplace=True)
# ind = (stock_data5['return'] >= -0.1) & (stock_data5['return'] <= 0.1)
# stock_data5 = stock_data5.loc[ind, :]
# plt.subplot(2,2,1)
# stock_data5['return'].plot()
# merge_data = pd.merge(left=merge_data,right=stock_data5[['date','return']],
#                       on='date',
#                       how='inner')#第一个为股票第二为指数按照日期去拼
# merge_data.columns = ['date', 'return_708', 'rfreturn', 'return_指数','return_20','return_547','return_982','return2347']
# print(stk_ret)
# stk_ret = merge_data['return2347'].values
# print(merge_data)
# rf_ret = merge_data['rfreturn'].values
# ind_ret = merge_data['return_指数'].values
# plt.subplot(2,2,2)
# #2347单因子资产检验
# plt.plot(ind_ret - rf_ret, stk_ret - rf_ret, 'o', ms=5, mfc='w', lw=2)#股票收益减去无风险
# plt.xlabel(r'$r_m - r_f$', fontsize = 20)
# plt.ylabel(r'$r_i - r_f$', fontsize = 20)
# x = sm.add_constant(ind_ret-rf_ret)#定义x
# y = stk_ret - rf_ret
# model = sm.OLS(y, x)
# results = model.fit()
# print(results.summary())
# plt.show()

#
#
#
#
#
# #多因子#整理月度数据
stock_data = pd.read_csv('month data.csv',encoding='GB2312')
stock_data.dropna(inplace=True)
stock_data.columns = ['code','date', 'close', 'rfreturn']


# # 股票代码_Stkcd    日期_Date  收盘价_Clpr  日无风险收益率_DRfRet
stk_codes = np.unique(stock_data['code'].values)#共8个股票代码
print(stk_codes)
plt.figure(9)
plt.subplot(2,2,1)
stock_data50 = stock_data[stock_data['code'] == stk_codes[0]]
stock_data50['date'] = pd.to_datetime(stock_data50['date'])
stock_data50.sort_values(by=['date'], inplace=True)
stock_data50['return'] = np.log(stock_data50['close']) - np.log(stock_data50['close'].shift(periods=1))
stock_data50.dropna(inplace=True)
ind = (stock_data50['return'] >= -0.1) & (stock_data50['return'] <= 0.1)
stock_data50 = stock_data50.loc[ind, :]
print(stock_data50)
plt.plot(stock_data50['return'].values)
# plt.show()
plt.subplot(2,2,2)
stock_data51 = stock_data[stock_data['code'] == stk_codes[1]]
stock_data51['date'] = pd.to_datetime(stock_data51['date'])
stock_data51.sort_values(by=['date'], inplace=True)
stock_data51['return'] = np.log(stock_data51['close']) - np.log(stock_data51['close'].shift(periods=1))
stock_data51.dropna(inplace=True)
ind = (stock_data51['return'] >= -0.1) & (stock_data51['return'] <= 0.1)
stock_data51 = stock_data51.loc[ind, :]
plt.plot(stock_data51['return'].values)
plt.subplot(2,2,3)
stock_data52 = stock_data[stock_data['code'] == stk_codes[2]]
stock_data52['date'] = pd.to_datetime(stock_data52['date'])
stock_data52.sort_values(by=['date'], inplace=True)
stock_data52['return'] = np.log(stock_data52['close']) - np.log(stock_data52['close'].shift(periods=1))
stock_data52.dropna(inplace=True)
ind = (stock_data52['return'] >= -0.1) & (stock_data52['return'] <= 0.1)
stock_data52 = stock_data52.loc[ind, :]
plt.plot(stock_data52['return'].values)
plt.subplot(2,2,4)
stock_data53 = stock_data[stock_data['code'] == stk_codes[3]]
stock_data53['date'] = pd.to_datetime(stock_data53['date'])
stock_data53.sort_values(by=['date'], inplace=True)
stock_data53['return'] = np.log(stock_data53['close']) - np.log(stock_data53['close'].shift(periods=1))
stock_data53.dropna(inplace=True)
ind = (stock_data53['return'] >= -0.1) & (stock_data53['return'] <= 0.1)
stock_data53 = stock_data53.loc[ind, :]
plt.plot(stock_data53['return'].values)
# plt.show()
#
# plt.figure(8)
plt.subplot(2,2,1)
stock_data54 = stock_data[stock_data['code'] == stk_codes[4]]
stock_data54['date'] = pd.to_datetime(stock_data54['date'])
stock_data54.sort_values(by=['date'], inplace=True)
stock_data54['return'] = np.log(stock_data54['close']) - np.log(stock_data54['close'].shift(periods=1))
stock_data54.dropna(inplace=True)
ind = (stock_data54['return'] >= -0.1) & (stock_data54['return'] <= 0.1)
stock_data54 = stock_data54.loc[ind, :]
plt.plot(stock_data54['return'].values)

plt.subplot(2,2,2)
stock_data55 = stock_data[stock_data['code'] == stk_codes[5]]
stock_data55['date'] = pd.to_datetime(stock_data55['date'])
stock_data55.sort_values(by=['date'], inplace=True)
stock_data55['return'] = np.log(stock_data55['close']) - np.log(stock_data55['close'].shift(periods=1))
stock_data55.dropna(inplace=True)
ind = (stock_data55['return'] >= -0.1) & (stock_data55['return'] <= 0.1)
stock_data55 = stock_data55.loc[ind, :]
plt.plot(stock_data55['return'].values)

plt.subplot(2,2,3)
stock_data56 = stock_data[stock_data['code'] == stk_codes[6]]
stock_data56['date'] = pd.to_datetime(stock_data56['date'])
stock_data56.sort_values(by=['date'], inplace=True)
stock_data56['return'] = np.log(stock_data56['close']) - np.log(stock_data56['close'].shift(periods=1))
stock_data56.dropna(inplace=True)
ind = (stock_data56['return'] >= -0.1) & (stock_data56['return'] <= 0.1)
stock_data56 = stock_data56.loc[ind, :]
plt.plot(stock_data56['return'].values)

plt.subplot(2,2,4)
stock_data57 = stock_data[stock_data['code'] == stk_codes[7]]
stock_data57['date'] = pd.to_datetime(stock_data57['date'])
stock_data57.sort_values(by=['date'], inplace=True)
stock_data57['return'] = np.log(stock_data57['close']) - np.log(stock_data57['close'].shift(periods=1))
stock_data57.dropna(inplace=True)
ind = (stock_data57['return'] >= -0.1) & (stock_data57['return'] <= 0.1)
stock_data57 = stock_data57.loc[ind, :]
plt.plot(stock_data57['return'].values)
# plt.show()
print(stock_data)
#

#转为月度数据记得复习
month300=pd.read_csv('300dailydata.csv',encoding='GB2312')
month300['交易日期_TrdDt']=pd.to_datetime(month300['交易日期_TrdDt'])
month300.index=month300['交易日期_TrdDt']#转月度要将index设成日期
month300=month300.resample('M').last()
del month300['交易日期_TrdDt']
month300=month300.reset_index()
month300.to_csv('300month.csv',encoding='utf-8')
index=pd.read_csv('300month.csv')
index = index.iloc[:, 1:].copy()#删除第一列
print(index)
index['交易日期_TrdDt']=pd.to_datetime(index['交易日期_TrdDt'])
print(index)
# 在您提供的代码中，`index.iloc[:, 1:]` 选择了从第二列开始的所有列，并且不包括第一列。
# 因为 `:` 表示所有行，因此第一个冒号留空。这将返回一个新的 DataFrame，其中包含原始 DataFrame 的所有行和除第一列外的所有列。
index.columns=['date','code','close']
index['return'] = np.log(index['close']) - np.log(index['close'].shift(periods=1))
index.dropna(inplace=True)
ind = (index['return'] >= -0.1) & (index['return'] <= 0.1)
index = index.loc[ind, :]
index.columns=['date','code','close','return']
# # index的日期为 float64
print(index)
print(stock_data50)
data_matrix = pd.merge(left=index,
                       right=stock_data50[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
data_matrix = pd.merge(left=data_matrix,
                       right=stock_data51[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)

data_matrix = pd.merge(left=data_matrix,
                       right=stock_data52[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
data_matrix = pd.merge(left=data_matrix,
                       right=stock_data53[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
data_matrix = pd.merge(left=data_matrix,
                       right=stock_data54[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)

data_matrix = pd.merge(left=data_matrix,
                       right=stock_data55[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
data_matrix = pd.merge(left=data_matrix,
                       right=stock_data56[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
data_matrix = pd.merge(left=data_matrix,
                       right=stock_data57[['date', 'return']],
                      on='date',
                      how='inner',
                      sort=True)
print(data_matrix)
ret_rf = stock_data[['date', 'rfreturn']]
ret_rf.dropna(inplace=True)
ret_rf.drop_duplicates(inplace=True, subset=['date'])#删除重复记得复习
#ret_rf.drop_duplicates(subset=['year', 'month'], inplace=True)#按照多列
# subset : column label or sequence of labels, optional用来指定特定的列，默认所有列
ret_rf.sort_values(by=['date'], inplace=True)#根据日期排序
ret_rf['date'] = pd.to_datetime(ret_rf['date'])
#拼无风险收益
data_matrix = pd.merge(left=data_matrix,
                       right=ret_rf,
                      on='date',
                      how='inner',
                      sort=True)
data_matrix.drop_duplicates(inplace=True, subset=['date','code'])
data_matrix.reset_index(inplace=True,drop=True)
data_matrix.columns = ['date', 'code', 'close','300return', 'stk2', 'stk3', 'stk4', 'stk5', 'stk6', 'stk7','stk8','stk9','rf']
print(data_matrix)
data_matrix['300return'] = data_matrix['300return'] - data_matrix['rf']
data_matrix['stk2'] = data_matrix['stk2'] - data_matrix['rf']
data_matrix['stk3'] = data_matrix['stk3'] - data_matrix['rf']
data_matrix['stk4'] = data_matrix['stk4'] - data_matrix['rf']
data_matrix['stk5'] = data_matrix['stk5'] - data_matrix['rf']
data_matrix['stk6'] = data_matrix['stk6'] - data_matrix['rf']
data_matrix['stk7'] = data_matrix['stk7'] - data_matrix['rf']
data_matrix['stk8'] = data_matrix['stk8'] - data_matrix['rf']
data_matrix['stk9'] = data_matrix['stk9'] - data_matrix['rf']
print(data_matrix)
ret_ind = data_matrix['300return'].values#指数超额收益
T = len(ret_ind)#数据长度
print(T)
# T = 11
#多资产回归 x为指数 y为
mu_market = np.mean(ret_ind)#指数均值
sigma_market = np.sum((ret_ind - mu_market)**2)/T#sigma公式
ret_stocks = data_matrix[['stk2', 'stk3', 'stk4', 'stk5', 'stk6','stk7','stk8','stk9']].values#股票收益率
#先做单因子检验
x = sm.add_constant(ret_ind)
y = ret_stocks[:, 3]#随便取一个
model = sm.OLS(y,x)
results = model.fit()
print(results.summary())
#总结
# constant的p值很小，表示可以拒绝原假设，对于某资产不能被CAPM所支持

#开始多资产
# #无限制模型参数估计
x = np.ones((T, 2))#定义X一列是1一列是指数
x[:, 1] = ret_ind#指数收益率
y = ret_stocks#8只股票月度收益率
# a、β estimate计算公式
xTx = np.dot(np.transpose(x), x)#x转制乘以x
xTy = np.dot(np.transpose(x), y)#x转制乘以y
AB_hat = np.dot(np.linalg.inv(xTx), xTy)
print(AB_hat)
ALPHA = AB_hat[0]#第一位
print(ALPHA)
BETA = AB_hat[1]#第二位
RESD = y - np.dot(x, AB_hat)#残差Residual vector
COV = np.dot(np.transpose(RESD), RESD)/T#残差斜方差
invCOV = np.linalg.inv(COV)#逆矩阵

# # 限制模型参数估计#a=0同样步骤
xr = np.ones((T, 1))#一列
xr[:, 0] = ret_ind#指数收益没有1了
yr = ret_stocks#八个股票
xrTxr = np.dot(np.transpose(xr), xr)
xrTyr = np.dot(np.transpose(xr), yr)
ABr_hat = np.dot(np.linalg.inv(xrTxr), xrTyr)
RESDr = yr - np.dot(xr, ABr_hat)
COVr = np.dot(np.transpose(RESDr), RESDr)/T
invCOVr = np.linalg.inv(COVr)
#开始多因子检验
# #卡方分布#wald检验ppt24页
trans_ALPHA = np.ones((len(ALPHA), 1))#a转置竖的
trans_ALPHA[:, 0] = ALPHA
N=8
SWchi2 = T*(1/(1+mu_market**2/sigma_market))*np.dot(np.dot(ALPHA, invCOV), trans_ALPHA)
SWF = (T-N-1)/N*(1/(1+mu_market**2/sigma_market))*np.dot(np.dot(ALPHA, invCOV), trans_ALPHA)
pvalue_Wchi2 = 1 - chi2.cdf(SWchi2[0], N)
pvalue_WF = 1 - f.cdf(SWF[0], N, T-N-1)#累计概率
print('Wald 检验的 P 值=', format(pvalue_Wchi2))#Wald 检验的 P 值
print('F 检验的 P 值=',format(pvalue_WF))#F 检验的 P 值
# 而累积分布函数则是F分布函数的积分，表示随机变量小于等于某个值的概率。
# 因此，`f.cdf`函数的输入参数为F分布的自由度和F统计量的值，输出为这个F统计量小于等于这个值的概率（即P值）。
# 例如，若我们使用F分布的自由度为3和15，F统计量为2的值来计算累积分布函数，那么`f.cdf(2, 3, 15)`的输出结果为0.164。
# 这意味着在自由度为3和15的F分布下，F统计量小于等于2的概率为0.164。

#LR TEST检验
SLRchi2 = T*(np.log(np.linalg.det(COVr)) - np.log(np.linalg.det(COV)))
pvalue_SLRchi2 = 1 - chi2.cdf(SLRchi2, N)
print('LR 检验的 P 值=',format(pvalue_SLRchi2))#LR 检验的 P 值
# chi2.cdf(SLRchi2, N)计算卡方分布的累积分布函数（Cumulative Distribution Function，CDF）。
# `chi2.cdf`函数的输入参数为卡方分布的自由度（后）和卡方统计量的值(前)，输出为这个卡方统计量小于等于这个值的概率（即P值）。
# 这意味着在自由度为5的卡方分布下，卡方统计量小于等于10的概率为0.648。

# 1. `SWchi2`：用于计算卡方分布的统计量，其中 `T` 表示样本数量，`mu_market` 表示市场均值，`sigma_market` 表示市场标准差，`ALPHA` 表示 CAPM 模型中的 alpha 系数，`invCOV` 表示协方差矩阵的逆矩阵，`trans_ALPHA` 表示 alpha 系数的转置矩阵。
# 2. `SWF`：用于计算 Wald 检验的统计量，其中 `T` 表示样本数量，`N` 表示资产数量，`mu_market` 表示市场均值，`sigma_market` 表示市场标准差，`ALPHA` 表示 CAPM 模型中的 alpha 系数，`invCOV` 表示协方差矩阵的逆矩阵，`trans_ALPHA` 表示 alpha 系数的转置矩阵。
# 3. `pvalue_Wchi2`：用于计算卡方分布检验的 P 值，其中 `N` 表示资产数量，`SWchi2` 表示卡方分布的统计量，`chi2.cdf()` 是累积分布函数，用于计算累积概率。
# 4. `pvalue_WF`：用于计算 Wald 检验的 P 值，其中 `N` 表示资产数量，`T` 表示样本数量，`SWF` 表示 Wald 检验的统计量，`f.cdf()` 是累积分布函数，用于计算累积概率。
#
# #似然检LM
a = np.zeros((8, 1))
a[:, 0] = np.sum(RESDr, axis=0)#限制模型残差项累加
salpha = np.dot(invCOVr, a)
b = np.dot(ret_ind, RESDr)
sbeta = np.zeros((8, 1))
sbeta[:, 0] = np.dot(invCOVr, b)
score = np.concatenate((salpha, sbeta), axis=0)#得分函数
a = np.concatenate((invCOVr*T, invCOVr*np.sum(ret_ind)), axis=1)#第一行
b = np.concatenate((invCOVr*np.sum(ret_ind), invCOVr*np.sum(ret_ind**2)), axis=1)#第二行
Minfo = np.concatenate((a, b), axis=0)#信息矩阵
SLMchi2 = np.dot(np.dot(np.transpose(score), np.linalg.inv(Minfo)), score)
pvalue_SLMchi2 = 1 - chi2.cdf(SLMchi2[0][0], N)
print('LM 检验的 P 值=',format(pvalue_SLMchi2))

print('{:>10s}, {:>10s}, {:>10s}, {:>10s}'.format('Wald Test1', 'F Test', 'LR Test', 'LM Test'))
print('{:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}'.format(SWchi2[0], SWF[0], SLRchi2, SLMchi2[0][0]))
print('{:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}'.format(pvalue_Wchi2, pvalue_WF, pvalue_SLRchi2, pvalue_SLMchi2))
#结论得出的结果不能拒绝原假设，检验的股票不存在超额回报率，多资产联合检验支持CAPM

# # 读取股票数据
# matrix = stock_data50
# matrix2= stock_data51
# matrix3= stock_data52
# matrix4= stock_data53
# matrix5= stock_data54
# print(matrix)
# print(matrix2)
# print(matrix3)
# print(matrix4)
# print(matrix5)
# matrix=pd.merge(left=matrix[['date','return']],right=matrix2[['date','return']],on='date',how='inner',sort=True)
# matrix.columns=['date','stk1','stk2']
# matrix=pd.merge(left=matrix,right=matrix3[['date','return']],on='date',how='inner',sort=True)
# matrix.columns=['date','stk1','stk2','stk3']
# matrix=pd.merge(left=matrix,right=matrix4[['date','return']],on='date',how='inner',sort=True)
# matrix.columns=['date','stk1','stk2','stk3','stk4']
# matrix=pd.merge(left=matrix,right=matrix5[['date','return']],on='date',how='inner',sort=True)
# matrix.columns=['date','stk1','stk2','stk3','stk4','stk5']
# print(matrix)

# 计算过去N个月的累积收益率
# 并计算持有这些投资组合M个月的累积收益率
# 遍历不同的N和M参数组合
# 累积收益率 = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
#
# # 遍历不同的N和M参数组合
# for N in [1, 3, 6, 12]:
#     for M in [1, 3, 6, 12]:
#         # 计算过去N个月的累积收益率
#         matrix['cum_return'] = matrix.iloc[:,1:].mean(axis=1).rolling(N).apply(lambda x: np.prod(1+x)-1, raw=True)
#
#         # 按照累积收益率由低到高排序，并分成5个等权重投资组合
#         matrix['group'] = pd.qcut(matrix['cum_return'], 5, labels=False)
#         matrix['group'] = matrix['group'] + 1
#         matrix['weight'] = 1 / 5
#         # 函数时，它返回的是每个数据所属的分组编号，从0开始编号。例如，如果将数据分成5组，则编号为0、1、2、3、4。在这里
#         # 我们将这些编号加1，从而将编号转换为1、2、3、4、5，这样更符合人们的习惯，也便于后续的分析和可视化。
#
#         # 计算投资组合的收益率
#         df_portfolio = matrix.groupby(['date', 'group'])['cum_return'].mean().reset_index()
#         print(df_portfolio)
#         df_portfolio = df_portfolio.pivot(index='date', columns='group', values='cum_return').fillna(0)
#         df_portfolio['portfolio'] = df_portfolio.mean(axis=1)#组合收益率
#         #每个收益率加1得到总回爆率
#         #np.cumprod为累计乘绩
#         df_portfolio['cum_return'] = np.cumprod(1+df_portfolio['portfolio'])-1
#
#         # 计算持有M个月的累积收益率
#         df_portfolio['cum_return_M'] = df_portfolio['cum_return'].rolling(M).apply(lambda x: np.prod(1+x)-1, raw=True)
#
#         # 判断市场惯性效应还是反转效应
#         if df_portfolio['cum_return_M'].iloc[-1] > 0:
#             print(f'N={N}, M={M}：市场存在惯性效应')
#         else:
#             print(f'N={N}, M={M}：市场存在反转效应')


#股市的反转效应与惯性效应代码
# 根据过去1、3、6、12月的收益率排序，构造5个等权重组合

# 重命名数据矩阵的列名

# 根据过去1、3、6、12月的收益率排序，构造5个等权重组合
# matrix= pd.melt(matrix, id_vars='date', var_name='code', value_name='return')
# # pd.melt 表示將原始數據集轉乘矩陣型式id_vars 保留的列,var_name表示转换后的列名，value_name表示转换后的值名
# print(matrix)
# matrix['ret_1m'] = matrix.groupby('code')['return'].apply(lambda x: x.pct_change())
# matrix['ret_3m'] = matrix.groupby('code')['return'].apply(lambda x: x.pct_change(3))
# matrix['ret_6m'] = matrix.groupby('code')['return'].apply(lambda x: x.pct_change(6))
# matrix['ret_12m'] = matrix.groupby('code')['return'].apply(lambda x: x.pct_change(12))
# # matrix['rank'] =matrix[['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m']].rank(method='average').sum(axis=1)
# matrix['rank']=matrix['ret_12m'].rank(method='average')
# matrix['group'] = pd.qcut(matrix['rank'], 5, labels=False)
# # pd.qcut表示将总排名分成五个等权重组合
# print(matrix)
# # #
# # # 计算每个组合的持有期收益率
# df_portfolio = matrix.groupby(['group', pd.Grouper(freq='M', key='date')])['return'].mean().reset_index()
# df_portfolio['cum_return_1m'] = df_portfolio.groupby('group')['return'].apply(lambda x: (1+x).cumprod()-1)
# df_portfolio['cum_return_3m'] = df_portfolio.groupby('group')['return'].apply(lambda x: (1+x).rolling(3).apply(np.prod, raw=True)-1)
# df_portfolio['cum_return_6m'] = df_portfolio.groupby('group')['return'].apply(lambda x: (1+x).rolling(6).apply(np.prod, raw=True)-1)
# df_portfolio['cum_return_12m'] = df_portfolio.groupby('group')['return'].apply(lambda x: (1+x).rolling(12).apply(np.prod, raw=True)-1)
# df_portfolio['cum_return_all'] = df_portfolio.groupby('group')['return'].apply(lambda x: (1+x).cumprod()-1)
# print(df_portfolio)
# # 使用`groupby`函数按照'group'和'Month'列计算每个组合在每个月的平均收益率，并将结果保存到新的'return'列中。
# # 然后，使用`reset_index`函数将多层索引转换为列，以便进行后续计算。接下来，使用`np.cumprod`函数计算每个组合在不同的持有期内的累计收益率，
# # 其中'cum_return_1m'、'cum_return_3m'、'cum_return_6m'和'cum_return_12m'分别表示持有期为1个月、3个月、6个月和12个月的累计收益率。
# # 最后，使用`np.cumprod`函数计算所有持有期的累计收益率，并将结果保存到'cum_return_all'列中。
# # 最终输出的结果为一个新的数据集'df_portfolio'，其中包含'group'、'Month'、'return'、'cum_return_1m'、'cum_return_3m'、'cum_return_6m'、'cum_return_12m'和'cum_return_all'这些列。
# # 其中，'group'和'Month'列分别表示等权重组合和持有期的标识；'return'列表示每个组合在每个持有期内的平均收益率；'cum_return_1m'、'cum_return_3m'、'cum_return_6m'和'cum_return_12m'列分别表示持有期为1个月、3个月、6个月和12个月的累计收益率；
# # 'cum_return_all'列表示所有持有期的累计收益率。
# # # 计算每个组合的平均持有期收益率
# mean_return_1m = df_portfolio['cum_return_1m'].mean()
# mean_return_3m = df_portfolio['cum_return_3m'].mean()
# mean_return_6m = df_portfolio['cum_return_6m'].mean()
# mean_return_12m = df_portfolio['cum_return_12m'].mean()
# mean_return_all = df_portfolio['cum_return_all'].mean()
#
# mean_return_winner=df_portfolio[df_portfolio['group']==0]['cum_return_12m'].mean()
# mean_return_loser=df_portfolio[df_portfolio['group']==4]['cum_return_12m'].mean()
# # # 判断市场的惯性效应或反转效应
# if mean_return_winner > mean_return_loser:
#     print("市场存在惯性效应")
# else:
#     print("市场存在反转效应")
# print(mean_return_winner)
# print(mean_return_loser)
#w=-0.10013613325263433,l=0.008132457102168944 反转
#w=-0.08742307934696318 l=0.0229114122422797
#w=-0.13585583769678802 l=-0.037825302914763964
#w=-0.03906055921577628 l=0.09138682809505645
#中，我们首先将数据矩阵的列名重命名为'date'和'stk1'-'stk5'，然后使用`pd.melt()`将数据矩阵转换为包含日期、
# 股票代码和股票收益率的dataframe。接下来，我们计算每个们比较1个月持有期的平均收益率和所有持有期的平均收益率的大小，以判断市场的惯性效应或反转效应。
# 如果1个月的平均收益率大于所有持有期的平均收益率，则表示市场存在惯性效应；反之，则表示市场存在反转效应。


# 从已经保存的文件中读取数据
data=pd.read_csv('市场数据1.csv',encoding='GB2312')
# for i in range(2, 5):
#     data_read = pd.read_csv('市场数据' + str(i) + '.csv', encoding='GB2312')
#     data = data.append(data_read, ignore_index=True)
#     del data_read
# data.drop(columns=['Unnamed: 3'],inplace=True)
# print(data)
# data.columns=['stk','date','ret']
# #读取且往右边拼(实验5）
# industry_li = np.unique(data['stk'].values)
# print(len(industry_li))#3632
# data2=data[data['stk'] == industry_li[0]]
# #重要循环拼数据将月收益率往右拼
# for i in range(1,500):
#     temp = data[data['stk'] == industry_li[i]]
#
#     data2 = pd.merge(left=data2,
#                      right=temp[['date', 'ret']],
#                      on=['date'],how='left')
#
# print(data2)

# 定义参数

returns_df=data
#
N_values = [1, 3, 6, 12]
M_values = [1, 3, 6, 12]
num_portfolios = 5

# 进行实证检验
results = {}
for N in N_values:
    for M in M_values:
        momentum_portfolio_returns = []
        for t in range(N + M, len(returns_df) - M):
            past_returns = returns_df.iloc[t - N:t].sum()
            past_returns_sorted = past_returns.sort_values()
            portfolio_returns = []
            group_size = len(past_returns_sorted) // num_portfolios
            for p in range(num_portfolios):
                group = past_returns_sorted.iloc[p * group_size: (p + 1) * group_size]
                group_return = returns_df.iloc[t + 1:t + M + 1][group.index].mean(axis=1).sum()
                portfolio_returns.append(group_return)
            momentum_portfolio_returns.append(portfolio_returns)
        results[(N, M)] = pd.DataFrame(momentum_portfolio_returns, columns=[f'P{i+1}' for i in range(num_portfolios)])

# 输出结果
for key, value in results.items():
    print(f"\nN = {key[0]}, M = {key[1]}")
    sorted_mean_returns = value.mean().sort_values(ascending=False)
    print(sorted_mean_returns)

    # 判断是惯性效应还是反转效应
    if sorted_mean_returns.index[-1] == 'P1' and sorted_mean_returns.index[0] == 'P5':
        print('Market exhibits a momentum effect.')
    elif sorted_mean_returns.index[-1] == 'P5' and sorted_mean_returns.index[0] == 'P1':
        print('Market exhibits a reversal effect.')
    else:
        print('Market does not exhibit a clear momentum or reversal effect.')

def plot_results(N, M, results):
    data = results[(N, M)].mean()
    data.plot(kind='bar')
    plt.title(f"N = {N}, M = {M}")
    plt.xlabel('Portfolio')
    plt.ylabel('Average Return')
    plt.show()

for N in N_values:
    for M in M_values:
        plot_results(N, M, results)
