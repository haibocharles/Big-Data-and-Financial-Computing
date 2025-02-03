import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, chi2, genpareto
import matplotlib.pyplot as plt
from arch import arch_model

# res = requests.get('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz000676&scale=240&ma=no&datalen=10000')
# # scale单位是分钟。这个地址数据很全，开盘、收盘、最高、最低、成交量。
# # ma 移动平均参数
# # datalen 数据量参数
# data_json = res.json()
# data = pd.DataFrame(data_json)
# data.to_csv('data_ssec.csv')#存数据
# print(data)
#收益率数据
data = pd.read_csv('data_ssec.csv')#读数据
data['return']=np.log(data['close'])-np.log(data['close'].shift(periods=1))
data['day'] = pd.to_datetime(data['day'], format='%Y-%m-%d')
print(data)
ind = data['day'] >= pd.to_datetime('2003-01-01', format='%Y-%m-%d')#样本外
r = data[ind]['return'].values*100
plt.plot(r)
plt.show()
# 假设样本内外数据量比为1/2，用样本内数据估计模型参数，再进行滚动窗口预测，滚动N次，在α=5%的前提下计算VaR，并画出VaR的N个预测值的时间序列图。
#参数法 RiskMetrics方法：假设收益率满足正态分布
l = np.fix(len(r)/3).astype(int)
VaR_RM = np.zeros(len(r))#0向量
qalpha = norm.ppf(0.05)#将临界值算出来za
#计算正态分布参数 mu和sigma
for i in range(l, len(r)):
    mhat, shat = norm.fit(r[i-200:i])#mu和sigma前面200天
    VaR_RM[i] = -(mhat + qalpha*shat)
plt.plot(r)
plt.plot(VaR_RM*-1)
print(VaR_RM)
plt.show()

#参数法-GARCH-normal
l = np.fix(len(r) / 3).astype(int)
VaR_GN = np.zeros(len(r))
qalpha = norm.ppf(0.05)
for i in range(l, len(r)):
    am_ar_garch = arch_model(r[:i], mean='ar', lags=1, vol='garch', dist='normal', p=2, q=2)#注意这里
    res_ar_garch = am_ar_garch.fit()
    a = res_ar_garch.forecast(horizon=1, align='origin')#预测未来一天的均值 mu sigma
    mu = a.mean['h.1'].iloc[-1]
    sigma = a.variance['h.1'].iloc[-1]
    VaR_GN[i] = -(mu + qalpha * np.sqrt(sigma))
plt.plot(r)
plt.plot(VaR_GN * -1)
plt.show()
# 历史模拟方法假定历史会重现，利用资产组合的收益率历史数据，在给定置信水平下，确定资产组合在持有期内的最低收益。
# 单资产：
# 给定资产收益率序列，计算t+1时刻的VaR
# 提取出200个历史收益率数据（t-200:t）
# 将历史收益率数据从小到大排序
# 在置信度为95%下，VaR为第5个收益率值
# 在NumPy中，`np.fix()`是一个函数，它返回最接近输入元素的整数并朝着零方向舍入。它接受一个参数并返回与输入相同形状的数组或标量。-3.2》3.2
l = np.fix(len(r)/3).astype(int)
VaR_HS = np.zeros(len(r))
qalpha = int(200*0.05)#提取200个
for i in range(l, len(r)):
    his_sample = r[i-200:i]
    his_sample = np.sort(his_sample)
    VaR_HS[i] = -his_sample[qalpha-1]
plt.plot(r)
plt.plot(VaR_HS*-1)
plt.show()

# # 极值理论 POT方法阈值设置u为样本内数据十分位对应的值
# 最小值为极端值
# ceil向上取整
l = np.fix(len(r)/3).astype(int)
VaR_EVT = np.zeros(len(r))#零向量
alpha = 0.05
for i in range(l, len(r)):
    his_sample = r[i-200:i]
    his_sample = np.sort(his_sample)
    ind = np.ceil(len(his_sample)*0.1).astype(int)
    evt_sample = np.abs(his_sample[:ind])
    u = evt_sample[-1]# 最小值为极端值
    evt_sample = evt_sample - u# 样本都减去u
    evt_sample = np.delete(evt_sample, -1)#删除u

    n = len(his_sample)
    Nu = len(evt_sample)

    parmhat = genpareto.fit(evt_sample, floc=0)#广义帕累托分布（GPD）
    kHat = parmhat[0]; # Tail index parameter
    sigmaHat = parmhat[2]; # Scale parameter
    VaR_EVT[i] = u + sigmaHat / kHat * ((alpha * n / Nu) ** -kHat - 1)#计算VAR
plt.plot(r)
plt.plot(VaR_EVT*-1)
plt.show()

#存储个VAR
# data = pd.DataFrame({'return': r, 'VaR_RM': VaR_RM, 'VaR_GN': VaR_GN, 'VaR_HS': VaR_HS, 'VaR_EVT': VaR_EVT})#四种模型预测的VAR放进dataframe里面
# data.to_csv('Data_VaR.csv')
print(data)
#Kupiec检验
#_Kupiec和Christoffersen的原假设为预测正确，诺拒绝原假设，可认为模型预测不准确
def myfun_Kupiec(r, VaR, pstar):
    N = np.sum(r > VaR)#超过记为负值
    T = len(r)#天数
    LRuc = -2*((T-N)*np.log(1-pstar)+N*np.log(pstar)) + 2*((T-N)*np.log(1-N/T)+N*np.log(N/T))#公式
    pvalue_LRuc = 1 - chi2.cdf(LRuc, 1)
    return LRuc, pvalue_LRuc

def myfun_Christoffersen(r, VaR):
    ind = r > VaR
    ind1 = ind[:-1]#取出除了最后一个元素以外的所有元素
    ind2 = ind[1:]#从第二个到最后一个
    n00 = np.sum((ind1==0) & (ind2==0))
    n01 = np.sum((ind1==0) & (ind2==1))
    n10 = np.sum((ind1==1) & (ind2==0))
    n11 = np.sum((ind1==1) & (ind2==1))
    # 若给定某天VaR 没被超出，则偏差指标It为0，否则为1
    Pi01 = n01/(n01+n00)
    Pi11 = n11/(n10+n11)
    Pi2 = (n01+n11)/(n00+n01+n10+n11)

    LRind = (n00+n10)*np.log(1-Pi2) + (n01+n11)*np.log(Pi2) - \
            n00*np.log(1-Pi01) - n01*np.log(Pi01) - n10*np.log(1-Pi11) - n11*np.log(Pi11)#公式
    LRind = LRind*-2
    pvalue_LRind = 1 - chi2.cdf(LRind, 1)
    return LRind, pvalue_LRind
#条件覆盖检验
def myfun_Kupiec_Christoffersen(LRuc, LRind):
    LRcc = LRuc + LRind
    pvalue_LRcc = 1 - chi2.cdf(LRcc, 2)
    return LRcc, pvalue_LRcc

data = pd.read_csv('Data_VaR.csv')
ind = data['VaR_RM'] > 0
r = data.loc[ind, ['return']].values*-1#收益率是负的要乘以负的
VaR_RM = data.loc[ind, ['VaR_RM']].values
VaR_GN = data.loc[ind, ['VaR_GN']].values
VaR_HS = data.loc[ind, ['VaR_HS']].values
VaR_EVT = data.loc[ind, ['VaR_EVT']].values

pstar = 0.05
#
[LRuc_RM, pvalue_LRuc_RM] = myfun_Kupiec(r, VaR_RM, pstar)
[LRind_RM, pvalue_LRind_RM] = myfun_Christoffersen(r, VaR_RM)
[LRcc_RM, pvalue_LRcc_RM] = myfun_Kupiec_Christoffersen(LRuc_RM, LRind_RM)

[LRuc_GN, pvalue_LRuc_GN] = myfun_Kupiec(r, VaR_GN, pstar)
[LRind_GN, pvalue_LRind_GN] = myfun_Christoffersen(r, VaR_GN)
[LRcc_GN, pvalue_LRcc_GN] = myfun_Kupiec_Christoffersen(LRuc_GN, LRind_GN)

[LRuc_HS, pvalue_LRuc_HS] = myfun_Kupiec(r, VaR_HS, pstar)
[LRind_HS, pvalue_LRind_HS] = myfun_Christoffersen(r, VaR_HS)
[LRcc_HS, pvalue_LRcc_HS] = myfun_Kupiec_Christoffersen(LRuc_HS, LRind_HS)

[LRuc_EVT, pvalue_LRuc_EVT] = myfun_Kupiec(r, VaR_EVT, pstar)
[LRind_EVT, pvalue_LRind_EVT] = myfun_Christoffersen(r, VaR_EVT)
[LRcc_EVT, pvalue_LRcc_EVT] = myfun_Kupiec_Christoffersen(LRuc_EVT, LRind_EVT)


print('{:12s}, {:>12s}, {:>12s}, {:>12s}, {:>12s}, {:>12s}, {:>12s}'.format('', 'LRuc', 'pLRuc', 'LRind', 'pLRind', 'LRcc', 'pLRcc'))
print('{:12s}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}'.format('RiskMetrics', LRuc_RM, pvalue_LRuc_RM, LRind_RM, pvalue_LRind_RM, LRcc_RM, pvalue_LRcc_RM))
print('{:12s}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}'.format('GarchNormal', LRuc_GN, pvalue_LRuc_GN, LRind_GN, pvalue_LRind_GN, LRcc_GN, pvalue_LRcc_GN))
print('{:12s}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}'.format('HisSim', LRuc_HS, pvalue_LRuc_HS, LRind_HS, pvalue_LRind_HS, LRcc_HS, pvalue_LRcc_HS))
print('{:12s}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}'.format('EVT GPD', LRuc_EVT, pvalue_LRuc_EVT, LRind_EVT, pvalue_LRind_EVT,LRcc_EVT, pvalue_LRcc_EVT))