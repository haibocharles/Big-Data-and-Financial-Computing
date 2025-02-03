import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import statsmodels as sm
from statsmodels.tsa.stattools import acf
import scipy.stats as stats
#循环读数据
data = pd.read_csv('两个股数据1.csv', encoding='GB2312', usecols=[0,1,2])
for i in range(2, 5):
    data_read = pd.read_csv('两个股数据' + str(i) + '.csv', encoding='GB2312', usecols=[0,1,2])
    data = data.append(data_read, ignore_index=True)
    del data_read
data.columns=['stkcd','date','close']
data2=pd.read_csv('深成指数据.csv', encoding='GB2312', usecols=[0,1,2])
data=data.append(data2, ignore_index=True)#拼到上面
data3 = pd.read_csv('1LShowData_SSEC_daily.csv ', encoding='GB2312', usecols=[0,2, 6])
data=data.append(data3, ignore_index=True)
print(data)
data.dropna(inplace=True)
data['date']=pd.to_datetime(data['date'],format='%Y-%m-%d')
stkcd=np.unique(data['stkcd'])
print(stkcd)
p1 = data[data['stkcd'] == stkcd[0]]['close'].values#重要
r1 = np.log(p1[1:]) - np.log(p1[:-1])#取第二个以后 取第一个到倒数第二个
plt.plot(r1)
p2 = data[data['stkcd'] == stkcd[1]]['close'].values
r2 = np.log(p2[1:]) - np.log(p2[:-1])
plt.plot(r2)
p3= data[data['stkcd'] == stkcd[2]]['close'].values
r3= np.log(p3[1:]) - np.log(p3[:-1])
plt.plot(r3)
p4 = data[data['stkcd'] == stkcd[3]]['close'].values
r4 = np.log(p4[1:]) - np.log(p4[:-1])
plt.plot(r4)
# 第一题对指数和个股收益率和收益率绝对值进行描述性统计（计算交易日天数、均值、标准差、偏度、峰度、最大值、最小值和自相关系数）
# 计算交易日天数
tradingday=[]
tradingday1=len(p1)
tradingday2=len(p2)
tradingday3=len(p3)
tradingday4=len(p4)
tradingday=[tradingday1,tradingday2,tradingday3,tradingday4]
print(tradingday)
# 计算均值
mean=[]
mean_r1 = np.mean(r1)
mean_r11= np.mean(abs(r1))
mean_r2 = np.mean(r2)
mean_r22= np.mean(abs(r2))
mean_r3 = np.mean(r3)
mean_r33= np.mean(abs(r3))
mean_r4 = np.mean(r4)
mean_r44= np.mean(abs(r4))
mean=[mean_r1,mean_r2,mean_r3,mean_r4]
meanabs=[mean_r11,mean_r22,mean_r33,mean_r44]
print('\n')
print(mean)
print(meanabs)
# # 计算标准差
str_r=[]
std_r1 = np.std(r1)
std_r11 = np.std(abs(r1))
std_r2 = np.std(r2)
std_r22 = np.std(abs(r2))
std_r3 = np.std(r3)
std_r33 = np.std(abs(r3))
std_r4 = np.std(r4)
std_r44= np.std(abs(r4))
str=[std_r1,std_r2,std_r3,std_r4]
strr=[std_r11,std_r22,std_r33,std_r44]
print('\n')
print(str)
print(strr)
# # 计算偏度
a=stats.skew(r1)
print(a)
skew=[]
skew_r1 = np.mean((r1 - mean_r1) ** 3) / std_r1 ** 3
skew_r11=np.mean((abs(r1) - abs(mean_r11)) ** 3) / std_r1 ** 3
skew_r2 = np.mean((r2 - mean_r2) ** 3) / std_r2 ** 3
skew_r22=np.mean((abs(r2) - abs(mean_r22)) ** 3) / std_r1 ** 3
skew_r3 = np.mean((r3 - mean_r3) ** 3) / std_r3 ** 3
skew_r33=np.mean((abs(r3) - abs(mean_r33)) ** 3) / std_r1 ** 3
skew_r4 = np.mean((r4 - mean_r4) ** 3) / std_r4 ** 3
skew_r44=np.mean((abs(r4) - abs(mean_r44)) ** 3) / std_r1 ** 3
skew=[skew_r1,skew_r2,skew_r3,skew_r4]
skew2=[skew_r11,skew_r22,skew_r33,skew_r44]
print('\n')
print(skew)
print(skew2)

# # 计算峰度
a1=stats.kurtosis(r1)
print(a1)
kurt=[]
kurt_r1 = np.mean((r1 - mean_r1) ** 4) / std_r1 ** 4 - 3
kurt_r11 = np.mean((abs(r1) - mean_r11) ** 4) / std_r11 ** 4 - 3
kurt_r2 = np.mean((r2 - mean_r2) ** 4) / std_r2 ** 4 - 3
kurt_r22 = np.mean((abs(r2) - mean_r22) ** 4) / std_r22 ** 4 - 3
kurt_r3 = np.mean((r3 - mean_r3) ** 4) / std_r3 ** 4 - 3
kurt_r33 = np.mean((abs(r3) - mean_r33) ** 4) / std_r33 ** 4 - 3
kurt_r4 = np.mean((r4 - mean_r4) ** 4) / std_r4 ** 4 - 3
kurt_r44 = np.mean((abs(r4) - mean_r44) ** 4) / std_r44 ** 4 - 3
kurt=[kurt_r1,kurt_r2,kurt_r3,kurt_r4]
kurtabs=[kurt_r11,kurt_r22,kurt_r33,kurt_r44]
data_test=r1  # # 定义测试数据
#峰度
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0;
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('peakedness1 =',peakedness(data_test))
data_test=r2  # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('peakedness2 =',peakedness(data_test))
data_test=r3  # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('peakedness3 =',peakedness(data_test))
data_test=r4 # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('peakedness4 =',peakedness(data_test))
data_test=abs(r1)  # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0;
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('abspeakedness1 =',peakedness(data_test))
data_test=abs(r2)  # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0;
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('abspeakedness2 =',peakedness(data_test))
data_test=abs(r3)  # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0;
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('abspeakedness3 =',peakedness(data_test))
data_test=abs(r4) # # 定义测试数据
def peakedness(data):
    n = len(data) #样本个数
    average=np.mean(data) #计算平均值
    k1=n*(n+1)*(n-1)/(n-2)/(n-3)
    k2=3*(n-1)**2/(n-2)/(n-3)
    m2=0;
    m4=0
    for i in data:
        m4+=(i-average)**4
        m2+=(i-average)**2
    peakedness=k1*m4/m2**2-k2
    return peakedness
print('abspeakedness4 =',peakedness(data_test))
print('\n')
print(kurt)
print(kurtabs)
# # 计算最大值和最小值
max=[]
max_r1 = np.max(r1)
max_r11 = np.max(abs(r1))
max_r2 = np.max(r2)
max_r22 = np.max(abs(r2))
max_r3 = np.max(r3)
max_r33 = np.max(abs(r3))
max_r4 = np.max(r4)
max_r44 = np.max(abs(r4))
max=[max_r1,max_r2,max_r3,max_r4]
maxabs=[max_r11,max_r22,max_r33,max_r44]
print('\n')
print(max)
print(maxabs)
min=[]
min_r1 = np.min(r1)
min_r11 = np.min(abs(r1))
min_r2 = np.min(r2)
min_r22 = np.min(abs(r2))
min_r3 = np.min(r3)
min_r33 = np.min(abs(r3))
min_r4 = np.min(r4)
min_r44 = np.min(abs(r4))
min=[min_r1,min_r2,min_r3,min_r4]
minabs=[min_r11,min_r22,min_r33,min_r44]
print('\n')
print(min)
print(minabs)



matrix=data[data['stkcd'] == stkcd[0]]
matrix['r']=np.log(matrix['close'])-np.log(matrix['close'].shift(1))
print(matrix)
matrix.dropna(inplace=True)
matrix1=data[data['stkcd'] == stkcd[1]]
matrix1['r']=np.log(matrix1['close'])-np.log(matrix1['close'].shift(1))
print(matrix1)
matrix1.dropna(inplace=True)
matrix2=data[data['stkcd'] == stkcd[2]]
matrix2['r']=np.log(matrix2['close'])-np.log(matrix2['close'].shift(1))
print(matrix2)
matrix2.dropna(inplace=True)
matrix3=data[data['stkcd'] == stkcd[3]]
matrix3['r']=np.log(matrix3['close'])-np.log(matrix3['close'].shift(1))
print(matrix3)
matrix3.dropna(inplace=True)
matrix=pd.merge(left=matrix[['date','r']],right=matrix1[['date','r']],how='inner',on='date',sort=True)
matrix=pd.merge(left=matrix,right=matrix2[['date','r']],how='inner',on='date',sort=True)
matrix=pd.merge(left=matrix,right=matrix3[['date','r']],how='inner',on='date',sort=True)
matrix.columns=['date','1','5','676','399001']
print(matrix)
del matrix['date']
#自相关系数转成往右边拼接
print(matrix.corr())
print(abs(matrix).corr())




#garch
# # mu常数，波动率 GARCH(1, 1)，分布 正态
am_garch1 = arch_model(r1, mean='constant', vol='garch', p=1, q=1, dist='normal')
res_garch1 = am_garch1.fit()
print(res_garch1.summary())
paramgarch1=res_garch1.params
print(paramgarch1)
am_garch2 = arch_model(r2, mean='constant', vol='garch', p=1, q=1, dist='normal')
res_garch2 = am_garch2.fit()
print(res_garch2.summary())
paramgarch2=res_garch2.params
print(paramgarch2)
am_garch3= arch_model(r3, mean='constant', vol='garch', p=1, q=1, dist='normal')
res_garch3 = am_garch3.fit()
print(res_garch3.summary())
paramgarch3=res_garch3.params
print(paramgarch3)
am_garch4 = arch_model(r4, mean='constant', vol='garch', p=1, q=1, dist='normal')
res_garch4 = am_garch4.fit()
print(res_garch4.summary())
paramgarch4=res_garch4.params
print(paramgarch4)

arctest1 = het_arch(res_garch1.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest1[0],arctest1[1],arctest1[2],arctest1[3]))
arctest2 = het_arch(res_garch2.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest2[0],arctest2[1],arctest2[2],arctest2[3]))
arctest3 = het_arch(res_garch3.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest3[0],arctest3[1],arctest3[2],arctest3[3]))
arctest4 = het_arch(res_garch4.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest4[0],arctest4[1],arctest4[2],arctest4[3]))



# # mu AR，波动率 GARCH(1, 1)，分布 正态 均值模型变成ar ar(1)
am_ar_garch1 = arch_model(r1, mean='constant', vol='arch', p=1, q=1, dist='normal')
res_am_ar_garch1 = am_ar_garch1.fit()
print(res_am_ar_garch1.summary())
paramarch1=res_am_ar_garch1.params
print(paramarch1)
# aEb 或者 aeb(其中a是浮点数，b是整数)表示a乘以10的b次方。例如：e+26 =10^26，1e+1=10。
am_ar_garch2 = arch_model(r2, mean='constant', vol='arch', p=1, q=1, dist='normal')
res_am_ar_garch2 = am_ar_garch2.fit()
print(res_am_ar_garch2.summary())
paramarch2=res_am_ar_garch1.params
print(paramarch2)
am_ar_garch3 = arch_model(r3, mean='constant', vol='arch', p=1, q=1, dist='normal')
res_am_ar_garch3 = am_ar_garch3.fit()
print(res_am_ar_garch3.summary())
paramarch3=res_am_ar_garch3.params
print(paramarch3)
am_ar_garch4 = arch_model(r4, mean='constant', vol='arch', p=1, q=1, dist='normal')
res_am_ar_garch4 = am_ar_garch4.fit()
print(res_am_ar_garch4.summary())
paramarch4=res_am_ar_garch4.params
print(paramarch4)
# 对残差序列进行ARCH效应和异方差性检验
# 对残差序列进行处理，删除缺失值和无穷大的值
#mean=ar 就是arma（1）

# 对残差序列进行处理，删除缺失值和无穷大的值
resid = res_am_ar_garch1.resid
resid = resid[np.isfinite(resid)]

# 对残差序列进行ARCH效应和异方差性检验
LMstat, LM_pvalue, Fstat, f_value = het_arch(resid)
print('第一个ARCH模型的ARCH效应和异方差性检验结果：')
print('第一个LM检验的值{:.4f},LM检验的p值{:.4f},f检验的值{:.4f},f检验的p值{:.4f}'.format(LMstat, LM_pvalue, Fstat, f_value))
#第二个
resid2 = res_am_ar_garch2.resid
resid2 = resid2[np.isfinite(resid2)]

# 对残差序列进行ARCH效应和异方差性检验
print(het_arch(resid2))
LMstat, LM_pvalue, Fstat, f_value = het_arch(resid2)
print('第二个ARCH模型的ARCH效应和异方差性检验结果：')
print('第二个LM检验的值{:.4f},LM检验的p值{:.4f},f检验的值{:.4f},f检验的p值{:.4f}'.format(LMstat, LM_pvalue, Fstat, f_value))

#第三个
resid3 = res_am_ar_garch3.resid
resid3 = resid3[np.isfinite(resid3)]

# 对残差序列进行ARCH效应和异方差性检验
LMstat, LM_pvalue, Fstat, f_value= het_arch(resid3)
print('第三个ARCH模型的ARCH效应和异方差性检验结果：')
print('第三个LM检验的值{:.4f},LM检验的p值{:.4f},f检验的值{:.4f},f检验的p值{:.4f}'.format(LMstat, LM_pvalue, Fstat, f_value))

#第四个
resid4 = res_am_ar_garch4.resid
resid4 = resid4[np.isfinite(resid4)]

# 对残差序列进行ARCH效应和异方差性检验
LMstat, LM_pvalue, Fstat, f_value= het_arch(resid4)
print('第四个ARCH模型的ARCH效应和异方差性检验结果：')
print('第四个LM检验的值{:.4f},LM检验的p值{:.4f},f检验的值{:.4f},f检验的p值{:.4f}'.format(LMstat, LM_pvalue, Fstat, f_value))







#第四题EGARCH
am_egarch1 = arch_model(r1*100, mean='constant', vol='egarch', p=1, q=1, o=1)#p可能后
res_egarch1 = am_egarch1.fit()
print(res_egarch1.summary())
paramegarch1=res_egarch1.params
print(paramegarch1)
am_egarch2 = arch_model(r2*100, mean='constant', vol='egarch', p=1, q=1, o=1)
res_egarch2 = am_egarch2.fit()
print(res_egarch2.summary())
paramegarch2=res_garch2.params
print(paramegarch2)
am_egarch3 = arch_model(r3*100, mean='constant', vol='egarch', p=1, q=1, o=1)
res_egarch3 = am_egarch3.fit()
print(res_egarch3.summary())
paramegarch3=res_garch3.params
print(paramegarch3)
am_egarch4 = arch_model(r4*100, mean='constant', vol='egarch', p=1, q=1, o=1)
res_egarch4 = am_egarch4.fit()
print(res_egarch4.summary())
paramegarch4=res_garch4.params
print(paramegarch4)
#检验EGARCH(1, 1)模型能否刻画收益率的条件异方差特性。
import arch

# Egarch模型残差进行arch效应
arctest1 = het_arch(res_egarch1.resid,nlags=50)
print('对egarch模型的残差项做arthtest')
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest1[0],arctest1[1],arctest1[2],arctest1[3]))
arctest2 = het_arch(res_egarch2.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest2[0],arctest2[1],arctest2[2],arctest2[3]))
arctest3 = het_arch(res_egarch3.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest3[0],arctest3[1],arctest3[2],arctest3[3]))
arctest4 = het_arch(res_egarch4.resid,nlags=50)
print('LM值为={:.4f},LM值的p值为={:.4f},F值为={:.4f},F值的p值为={:.4f}'.format(arctest4[0],arctest4[1],arctest4[2],arctest4[3]))





#第五题
# 模型对指数和个股收益率进行模拟（模拟序列长度和原始收益率序列长度一样）
n_slim=len(matrix)
# sim_garch = arch_model(None, mean='constant', vol='garch', dist='normal', p=1, q=1)
# # print(res_garch.params)
# sim_paras=pd.Series([0.000184, 0.000006, 0.1, 0.88], index=['mu', 'omega', 'alpha[1]', 'beta[1]'])
# sim_garch_data = sim_garch.simulate(sim_paras, 1000)
# plt.plot(sim_garch_data)
plt.figure(1)#GARCH
plt.subplot(2,2,1)
sim_garch1 =am_garch1.simulate(paramgarch1, n_slim)
print(paramgarch1.shape)
print(sim_garch1)


plt.plot(matrix['1'],label='000001 Return',color='b')
plt.plot(matrix.index, sim_garch1 , label='GARCH(1) Simulated Returns')
plt.subplot(2,2,2)
sim_garch2 =am_garch2.simulate(paramgarch2, n_slim)
plt.plot(matrix['5'],label='000005 Return',color='b')
plt.plot(matrix.index, sim_garch2 , label='GARCH(2) Simulated Returns')
plt.subplot(2,2,3)
sim_garch3 =am_garch3.simulate(paramgarch3, n_slim)
plt.plot(matrix['676'],label='0006776 Return',color='b')
plt.plot(matrix.index, sim_garch3 , label='GARCH(3) Simulated Returns')
plt.subplot(2,2,4)
sim_garch4 =am_garch4.simulate(paramgarch4, n_slim)
plt.plot(matrix['399001'],label='399001 Return',color='b')
plt.plot(matrix.index, sim_garch4 , label='GARCH(4) Simulated Returns')
# plt.show()

plt.figure(2)#arch
plt.subplot(2,2,1)
sim_arch1 =am_ar_garch1.simulate(paramarch1, n_slim)
plt.plot(matrix['1'],label='000001 Return',color='b')
plt.plot(matrix.index, sim_arch1 , label='ARCH(1) Simulated Returns')
plt.subplot(2,2,2)
sim_arch2 =am_ar_garch2.simulate(paramarch2, n_slim)
plt.plot(matrix['5'],label='000005 Return',color='b')
plt.plot(matrix.index, sim_arch2 , label='ARCH(2) Simulated Returns')
plt.subplot(2,2,3)
sim_arch3 =am_ar_garch3.simulate(paramarch3, n_slim)
plt.plot(matrix['676'],label='000676 Return',color='b')
plt.plot(matrix.index, sim_arch3 , label='ARCH(3) Simulated Returns')
plt.subplot(2,2,4)
sim_arch4 =am_ar_garch4.simulate(paramarch4, n_slim)
plt.plot(matrix['399001'],label='399001 Return',color='b')
plt.plot(matrix.index, sim_arch4 , label='ARCH(4) Simulated Returns')
# plt.show()

plt.figure(3)#EGARCH
plt.subplot(2,2,1)
sim_egarch1 =am_egarch1.simulate(paramegarch1, n_slim)
plt.plot(matrix['1'],label='000001 Return',color='b')
plt.plot(matrix.index, sim_egarch1 , label='EGARCH(1) Simulated Returns')
plt.subplot(2,2,2)
sim_egarch2 =am_garch2.simulate(paramegarch2, n_slim)
plt.plot(matrix['5'],label='000005 Return',color='b')
plt.plot(matrix.index, sim_egarch2 , label='EGARCH(2) Simulated Returns')
plt.subplot(2,2,3)
sim_egarch3 =am_garch3.simulate(paramegarch3, n_slim)
plt.plot(matrix['676'],label='000676 Return',color='b')
plt.plot(matrix.index, sim_egarch3 , label='EGARCH(3) Simulated Returns')
plt.subplot(2,2,4)
sim_egarch4 =am_garch4.simulate(paramegarch4, n_slim)
plt.plot(matrix['399001'],label='399001 Return',color='b')
plt.plot(matrix.index, sim_egarch4 , label='EGARCH(4) Simulated Returns')
# plt.show()
