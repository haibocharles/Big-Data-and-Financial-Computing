import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import pi
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import warnings
from scipy.stats import f
warnings.filterwarnings('ignore')
#ooooo1
data = pd.read_csv('指数数据.csv', encoding='GB2312')#指数日度数据
data['交易日期_TrdDt']=pd.to_datetime(data['交易日期_TrdDt'])
data.set_index('交易日期_TrdDt', inplace=True)#记得设置index
data=data.resample('M').agg({'收盘价(元/点)_ClPr': 'last'})#转为月度数据
# data.resample('M').last(),记得要把代码删除
data['日期']=data.index
print(data)
# 证监会行业门类代码_Csrciccd1，证监会行业门类名称_Csrcicnm1，日期_Date，月交易天数_Montrds，行业月收益率_等权_Mreteq
data2 = pd.read_csv('实验5行业数据.csv', encoding='GB2312',usecols=[0,1,2,4])
data2.columns=['证监会行业门类代码_Csrciccd1', '证监会行业门类名称_Csrcicnm1', '日期','行业月收益率_等权_Mreteq']
data2['日期']=pd.to_datetime(data2['日期'])
print(data2)
industry_li = np.unique(data2['证监会行业门类代码_Csrciccd1'].values)
# print(data2)
#重要循环拼数据将月收益率往右拼
for i in range(10):
    temp = data2[data2['证监会行业门类代码_Csrciccd1'] == industry_li[i]]
    data = pd.merge(left=data,
                    right=temp[['日期', '行业月收益率_等权_Mreteq']],
                    on=['日期'],
                    how='inner')
print(data)
# data.to_excel('shuju.xlsx')
data3 = pd.read_csv('月CAPM因子.csv', encoding='GB2312')
# 日期_Date	市场溢酬因子__流通市值加权_Rmrf_tmv	市值因子__流通市值加权_Smb_tmv	账面市值比因子__流通市值加权_Hml_tmv
print(data3)
data3.columns=['日期','市场溢酬因子','市值因子','账面市值比因子']
data3['日期']=pd.to_datetime(data3['日期'])
matrix=pd.merge(left=data,right=data3[['日期','市场溢酬因子','市值因子','账面市值比因子']],on=['日期'],how='inner',sort=False)
print(matrix)
matrix.drop_duplicates(subset=['日期'],inplace=True)
data4=pd.read_csv('月无风险收益.csv', encoding='GB2312',usecols=[2,3])
data4.columns=['日期','月无风险收益率']
data4['日期']=pd.to_datetime(data4['日期'])
print(data4)
matrix=pd.merge(left=matrix,right=data4[['日期','月无风险收益率']],on=['日期'],how='inner',sort=False)
print(matrix)
matrix.columns=['市场指数', '日期', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', '市场溢酬因子','市值因子','账面市值比因子','rf']
matrix= matrix.dropna()
print(matrix)
# matrix.to_excel('资料.xlsx')
#以上清洗数据

p1 = matrix.iloc[:, 2].values#组合1
p2 = matrix.iloc[:, 3].values#组合2
p3 = matrix.iloc[:, 4].values#组合3
p4 = matrix.iloc[:, 5].values#组合4
p5 = matrix.iloc[:, 6].values#组合5
p6 = matrix.iloc[:, 7].values#组合6
p7= matrix.iloc[:, 8].values#组合7
p8= matrix.iloc[:, 9].values#组合8
p9= matrix.iloc[:, 10].values#组合9
p10= matrix.iloc[:, 11].values#组合10
rf = matrix.iloc[:, 15].values#无风险收益
mkt = matrix.iloc[:, 12].values#市场溢酬因子
smb=matrix.iloc[:, 13].values#市值因子
hml=matrix.iloc[:, 14].values#账面市值比因子
#算收益率
# np.log(p1[1:]) 表示取 p1 数组中第二个元素到最后一个元素的自然对数，而 np.log(p1[:-1]) 则表示取 p1 数组中第一个元素到倒数第二个元素的自然对数。
# 这两个结果都是长度为 n-1 的数组，其中 n 是 p1 数组的长度。然后，通过对这两个数组进行减法操作，得到长度为 n-1 的数组，
# 表示 p1 数组中每个相邻元素的自然对数值之间的差值。
# np.nanmean() 函数计算了该数组的平均值，忽略可能存在的任何 `NaN`（非数字）值。如果数组中存在 `NaN` 值，则使用 `np.isnan()` 函数和数组索引将其替换为计算出的平均值。
# 将 `NaN` 值替换为平均值的目的是为了避免由于缺失数据而导致进一步计算中出现错误。但是需要注意的是，这种方法假定缺失值是随机缺失的，而不是代表数据中的任何系统偏差。
#列表的计算
r1 = np.log(p1[1:]) - np.log(p1[:-1])
mean_val = np.nanmean(r1)
r1[np.isnan(r1)] = mean_val#将nan当作平均值
r2 = np.log(p2[1:]) - np.log(p2[:-1])
mean_val = np.nanmean(r2)
r2[np.isnan(r2)] = mean_val
r3 = np.log(p3[1:]) - np.log(p3[:-1])
mean_val = np.nanmean(r3)
r3[np.isnan(r3)] = mean_val
r4 = np.log(p4[1:]) - np.log(p4[:-1])
mean_val = np.nanmean(r4)
r4[np.isnan(r4)] = mean_val
r5 = np.log(p5[1:]) - np.log(p5[:-1])
mean_val = np.nanmean(r5)
r5[np.isnan(r5)] = mean_val
r6 = np.log(p6[1:]) - np.log(p6[:-1])
mean_val = np.nanmean(r6)
r6[np.isnan(r6)] = mean_val
r7 = np.log(p7[1:]) - np.log(p7[:-1])
mean_val = np.nanmean(r7)
r7[np.isnan(r7)] = mean_val
r8= np.log(p8[1:]) - np.log(p8[:-1])
# 数组中存在太多的 NaN 值，导致平均值也是 NaN。在这种情况下，你需要检查数组中的数据，并确保存在足够的非 NaN 值来计算平均值
# 代码的下一行将所有的 `NaN` 值替换为一个特定的值 `0.53272295`。这个值是手动设置的，你可以根据具体情况选择一个合适的值进行替换。
# 最后一行代码使用 `np.where()` 函数进一步处理 `r8` 数组，将其中所有的正无穷值和负无穷值替换为之前设置的特定值 `0.53272295`。这样做是为了避免这些特殊值对后续计算造成影响，确保数组中所有的值都是有限的。
r8[np.isnan(r8)] =0.53272295
r8 = np.where(np.isinf(r8), 0.53272295, r8)
r9= np.log(p9[1:]) - np.log(p9[:-1])
mean_val = np.nanmean(r9)
r9[np.isnan(r9)] = mean_val
r10= np.log(p10[1:]) - np.log(p10[:-1])
mean_val = np.nanmean(r10)
r10[np.isnan(r10)] = mean_val
#计算超额收益
rexc1 = r1 - rf[1:]
mean_val = np.nanmean(rexc1)
rexc1[np.isnan(rexc1)] = mean_val
rexc2 = r2 - rf[1:]
mean_val = np.nanmean(rexc2)
rexc2[np.isnan(rexc2)] = mean_val
rexc3 = r3 - rf[1:]
mean_val = np.nanmean(rexc3)
rexc3[np.isnan(rexc3)] = mean_val
rexc4 = r4 - rf[1:]
mean_val = np.nanmean(rexc4)
rexc4[np.isnan(rexc4)] = mean_val
rexc5 = r5 - rf[1:]
mean_val = np.nanmean(rexc5)
rexc5[np.isnan(rexc5)] = mean_val
rexc6 = r6 - rf[1:]
mean_val = np.nanmean(rexc6)
rexc6[np.isnan(rexc6)] = mean_val
rexc7 = r7 - rf[1:]
mean_val = np.nanmean(rexc7)
rexc7[np.isnan(rexc7)] = mean_val
rexc8 = r8- rf[1:]
mean_val = np.nanmean(rexc8)
rexc8[np.isnan(rexc8)] = mean_val
rexc9 = r9 - rf[1:]
mean_val = np.nanmean(rexc9)
rexc9[np.isnan(rexc9)] = mean_val
rexc10 = r10 - rf[1:]#减去第二个开始到最后一个
mean_val = np.nanmean(rexc10)
rexc10[np.isnan(rexc10)] = mean_val
# #计算样本方差斜方差矩阵#将不同向量拼起来
# np.concatenate()` 函数将这些处理过的数组拼接在一起，形成一个新的数组 `R`。这个函数将这些数组按列进行拼接，其中 `axis=1` 参数表示按列进行拼接。
# 在拼接过程中，每个数组都被转换成一个列向量，并且所有的列向量被垂直地拼接在一起。最终得到的数组 `R` 的行数为所有数组的行数之和，列数为数组的个数。
R = np.concatenate([rexc1[:, None],#将向量竖起来
                    rexc2[:, None],
                    rexc3[:, None],
                    rexc4[:, None],
                    rexc5[:, None],
                    rexc6[:, None],
                    rexc7[:, None],
                    rexc8[:, None],
                    rexc9[:, None],
                    rexc10[:, None]],
                   axis=1)
print(R)#数组
# #计算样本方差斜方差矩阵
# Cov_Sample = np.mat(np.cov(R, rowvar=False))#rowvar=False一列代表一个变量再转为矩阵
# print(Cov_Sample)#矩阵公式计算
# #因子模型估计法CAPM模型a +b +mkt
# X = np.mat(np.concatenate([np.ones((len(mkt)-1, 1)), mkt[1:, None]], axis=1))#常数项
# Y = np.mat(R)#记得转为mat
# AB_hat = (X.T*X).I*(X.T*Y)#X的转置的逆*X的转置*Y
# ALPHA = AB_hat[0]
# BETA = AB_hat[1]
# RESD = Y - X*AB_hat#残差
# covfactor = np.cov(mkt[1:])#因子协方差
# covresidual = np.diag(np.diag(np.cov(RESD, rowvar=False)))#残差协方差
# Cov_Factor = BETA.T*covfactor*BETA + covresidual
# print(Cov_Factor)#协方差矩阵
#
# #压缩估计法
# c = 0.5
# Cov_Shrink = c*Cov_Sample + (1-c)*Cov_Factor
# print(Cov_Shrink)
# uhat = np.mean(R, axis=0)
# #构建最优投资组合
# A = np.mat(np.concatenate([uhat[:, None], np.ones((len(uhat), 1))], axis=1)).T
# up = np.mean(uhat)
# b = np.mat(np.array([up, 1])[:, None])
# # b[np.isnan(b)] = 7.49300704e-05
# omega_Shrink = Cov_Shrink.I*A.T*(A*Cov_Shrink.I*A.T).I*b#最优投资组合压缩估计法
# print(omega_Shrink)
# omega_Sample = Cov_Sample.I*A.T*(A*Cov_Sample.I*A.T).I*b#样本方差协方差估计法
# print(omega_Sample)
# 第一种：样本模型
Cov_Sample=np.mat(np.cov(R,rowvar=False))#rowvar=False一列为一个因子
print('样本模型协方差矩阵为：',np.around(Cov_Sample,4))
#常量估计法
cov_1=np.cov(R,rowvar=False)
cov_chang=cov_1
sum_cov=0
duijiaoxian=np.zeros(10)
qitayuansu=np.zeros(100)#五个就是25？九个就是81？
for i in range(0,10):
    duijiaoxian[i]=cov_1[i][i]
for i in range(0,10):
    for j in range(0,10):
        sum_cov=cov_1[i][j]+sum_cov
sum_cov=sum_cov-np.sum(duijiaoxian)
avr_cov=sum_cov/95
duijiaoxian=np.average(duijiaoxian)
for i in range(0,10):
    for j in range(0,10):
        cov_chang[i][j]=avr_cov
for i in range(0, 5):
    cov_chang[i][i] = duijiaoxian
print('常量估计法的协方差矩阵为',np.around(cov_chang,5))
#第三种：因子模型FF
# X=np.mat(np.concatenate([np.ones((len(mkt)-1,1)),mkt[1:,None],smb[1:,None],hml[1:,None]],axis=1))#先转成matrix形式
# Y=np.mat(R)
# AB_hat=(X.T*X).I*(X.T*Y)#（X的转置乘以x的逆）*（X的转置乘以Y）
# ALPHA=AB_hat[0]
# BETA=AB_hat[1:]
# print(BETA)
# RESD=Y-X*AB_hat
# covfactor=np.cov([mkt[1:],smb[1:],hml[1:]])
# covresidual=np.diag(np.diag(np.cov(RESD,rowvar=False)))
# Cov_Factor=BETA.T*covfactor*BETA+covresidual
# print('因子模型的协方差矩阵为',np.around(Cov_Factor,4))
#因子模型估计法CAPM
X = np.mat(np.concatenate([np.ones((len(mkt)-1, 1)), mkt[1:, None]], axis=1))
Y = np.mat(R)
AB_hat = (X.T*X).I*(X.T*Y)
ALPHA = AB_hat[0]
BETA = AB_hat[1]
RESD = Y - X*AB_hat
covfactor = np.cov(mkt[1:])
covresidual = np.diag(np.diag(np.cov(RESD, rowvar=False)))
Cov_Factor = BETA.T*covfactor*BETA + covresidual
print('因子模型估计法的结果为：\n',Cov_Factor)
#常量估计法另解
diag_elements = np.diag(Cov_Factor)#因子矩阵对角线
mean_diag_elements = np.mean(diag_elements)#对角线平均值
upper_tri = np.triu(Cov_Factor,k=1)#上三角
lower_tri = np.tril(Cov_Factor,k=1)#下三角
mean_non_diag_element = (np.mean(upper_tri)+np.mean(lower_tri))/2
Con_Factor = np.ones((10,10))*mean_non_diag_element #五个就是5*5
np.fill_diagonal(Con_Factor,mean_diag_elements)
print('常量估计法的协方差矩阵为',np.around(Con_Factor,5))



#第四种 压缩估计法
n=10
T=len(R)
#不太懂下面这个
# id=np.ones([10,10])
# Id=pd.DataFrame(id)
# tr_cov=np.matrix.trace(Cov_Sample)
# tr_cov2=np.matrix.trace(Cov_Sample**2)
# pp=((1-2/n)*tr_cov2+tr_cov**2)/((T-2/n)*(tr_cov2-tr_cov**2/n))
# c=pp[0,0]
c=0.5
Cov_Shrink=(1-c)*Cov_Sample+c*Cov_Factor
print(c)
print('压缩估计法的协方差矩阵为',np.around(Cov_Shrink,4))
#
lamada=0.95
# m1=7*12+6
# m2=8*12+8
# m3=9*12+11
R_pd=pd.DataFrame(R)
#2018年是第64行
R_pd=R_pd.iloc[64:,:]#64行也就是2018年以后数据
print(R_pd)
rmat0=np.matrix(R_pd)
rmat1=rmat0[3:,:]#67
rmat2=rmat0[12:,:]#76
rmat3=rmat0[21:,:]#85
r0_avr=np.mean(rmat0,axis=0)
r1_avr=np.mean(rmat1,axis=0)
r2_avr=np.mean(rmat1,axis=0)
r3_avr=np.mean(rmat1,axis=0)
#因为为十个行业？
EWMA1=np.zeros([10,10])
EWMA2=np.zeros([10,10])
EWME3=np.zeros([10,10])
#指数加权估计法计算协方差矩阵
EWMA1=(1-lamada)*(rmat1-r1_avr).T*(rmat1-r1_avr)+lamada*Cov_Sample
EWMA2=(1-lamada)*(rmat2-r2_avr).T*(rmat2-r2_avr)+lamada*EWMA1
EWMA3=(1-lamada)*(rmat3-r3_avr).T*(rmat3-r3_avr)+lamada*EWMA2
print('2018年6月',np.around(EWMA1,4))
print('2019年8月',np.around(EWMA2,4))
print('2020年11月',np.around(EWMA3,4))
#指数加权另解（全部的）
#指数加权移动平均估计法EWMA
lamda = 0.95
n = Cov_Sample.shape[0]
ewma = Cov_Sample.copy()
for i in range(1,n):
    for j in range(i):
        ewma[i,j] = lamda*(ewma[i-1,j]+(1-lamda)*Cov_Sample[i,j])
        ewma[j,i] = ewma[i,j]
print('指数加权移动平均估计法的结果为：\n',ewma)
#第二题
#
# industries = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8",
#               "r9", "r10"]
# weights = pd.DataFrame(columns=industries)
#
# # 对于每个行业，计算其组合的权重
# for i in range(len(matrix)):
#     # 获取每月的数据
#     month_data = matrix.iloc[i, :]
#     # 对于每个行业，计算其组合的权重
#     month_weights = []
#     for industry in industries:
#         industry_data = month_data[industry]
#         X = np.mat(np.concatenate([np.ones((len(mkt) - 1, 1)), mkt[1:, None], smb[1:, None], hml[1:, None]], axis=1))
#         Y = np.mat(R)
#         AB_hat = (X.T * X).I * (X.T * Y)
#         ALPHA = AB_hat[0]
#         BETA = AB_hat[1:]
#         RESD = Y - X * AB_hat
#         covfactor = np.cov([mkt[1:], smb[1:], hml[1:]])
#         covresidual = np.diag(np.diag(np.cov(RESD, rowvar=False)))
#         Cov_Factor = BETA.T * covfactor * BETA + covresidual
#         uhat = np.mean(R, axis=0)
#         A = np.mat(np.concatenate([uhat[:, None], np.ones((len(uhat), 1))], axis=1)).T
#         up = np.mean(uhat)
#         b = np.mat(np.array([up, 1])[:, None])
#         omega_Sample = Cov_Factor.I * A.T * (A * Cov_Factor.I * A.T).I * b
#         month_weights.append(omega_Sample[1][0])
#         # 将该月的行业权重存储到权重矩阵中
#     weights.loc[i] = month_weights
#
#     # 输出每个月的行业权重变化
# print(weights)
# weights.to_excel('weights.xlsx')
# # 输出行业权重变化
date=matrix['日期']
date=pd.to_datetime(date)
print(date)

#因子模型估计收益率


# data=pd.merge(left=data_index[['yearmonth','nengyuan','cailiao','gongye','kexuan','xiaofei','yiyao','jinrong','xinxi','dianxin','gongyong']],
#                      right=data_factors[['yearmonth','mkt','smb','hml']],
#                      on=['yearmonth'],
#                      how='inner')
# print(data)
# data.columns=['yearmonth','nengyuan','cailiao','gongye','kexuan','xiaofei','yiyao','jinrong','xinxi','dianxin','gongyong','mkt','smb','hml']
# data.dropna(inplace=True)
# data=data.astype('float64')
#
# x=data.loc[:,['mkt','smb','hml']].values
#
# y1=data.loc[:,['nengyuan']].values
# y2=data.loc[:,['cailiao']].values
# y3=data.loc[:,['gongye']].values
# y4=data.loc[:,['kexuan']].values
# y5=data.loc[:,['xiaofei']].values
# y6=data.loc[:,['yiyao']].values
# y7=data.loc[:,['jinrong']].values
# y8=data.loc[:,['xinxi']].values
# y9=data.loc[:,['dianxin']].values
# y10=data.loc[:,['gongyong']].values





# def YYZHOU_LSQ(x,SMB,HML,y):
#     n=len(SMB)
#     sumx=np.sum(x)
#     sumSMB=np.sum(SMB)
#     sumHML=np.sum(HML)
#     sumx2=np.sum(x**2)
#     sumSMB2=np.sum(SMB**2)
#     sumHML2=np.sum(HML**2)
#     sumxSMB= np.sum(x*SMB)
#     sumxHML=np.sum(x*HML)
#     sumHMLSMB= np.sum(HML*SMB)
#     sumy= np.sum(y)
#     sumxy=np.sum(x*y)
#     sumHMLy= np.sum(HML*y)
#     sumSMBy = np.sum(SMB*y)
#     A = np.array([[n, sumx, sumSMB, sumHML], [sumx, sumx2, sumxSMB, sumxHML],
#                   [sumSMB, sumxSMB, sumSMB2, sumHMLSMB], [sumHML, sumxHML, sumHMLSMB, sumHML2]])
#     b = np.array([[sumy], [sumxy], [sumSMBy], [sumHMLy]])
#     parLSQ = np.dot(np.linalg.inv(A), b)
#     return parLSQ
#
# parLSQ1=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc1)
# parLSQ2=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc2)
# parLSQ3=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc3)
# parLSQ4=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc4)
# parLSQ5=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc5)
# parLSQ6=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc6)
# parLSQ7=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc7)
# parLSQ8=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc8)
# parLSQ9=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc9)
# parLSQ10=YYZHOU_LSQ(mkt[:],smb[:],hml[:],rexc10)
# parLSQ1=parLSQ1.reshape(4)
# parLSQ2=parLSQ2.reshape(4)
# parLSQ3=parLSQ3.reshape(4)
# parLSQ4=parLSQ4.reshape(4)
# parLSQ5=parLSQ5.reshape(4)
# parLSQ6=parLSQ6.reshape(4)
# parLSQ7=parLSQ7.reshape(4)
# parLSQ8=parLSQ8.reshape(4)
# parLSQ9=parLSQ9.reshape(4)
# parLSQ10=parLSQ10.reshape(4)
# # print('上证能源的最小二乘回归系数',parLSQ1)
# xishu=pd.DataFrame([parLSQ1,parLSQ2,parLSQ3,parLSQ4,parLSQ5,parLSQ6,parLSQ7,parLSQ8,parLSQ9,parLSQ10])
#xishu=np.mat(xishu.values)

factor=pd.DataFrame([np.ones(len(mkt)-1),mkt[1:],smb[1:],hml[1:]])
uhat = np.mean(R, axis=0)
mu = np.mean(uhat)
#mu.columns=['上证能源','上证材料','上证工业','上证可选','上证消费','上证医药','上证金融','上证信息','上证电信','上证公用']

#最小化期望方差权重
A=np.mat(np.concatenate([uhat[:,None],np.ones((len(uhat),1))],axis=1)).T
up=np.mean(uhat)
b=np.mat(np.array([up,1])[:,None])
# Cov_Factor=因子估计法
omega_factor1=Cov_Factor.I*A.T*(A*Cov_Factor.I*A.T).I*b
print('因子模型估计法在最小化风险条件下的权重',omega_factor1)
omega_factor1=omega_factor1.reshape(10)#改
omega_factor1=pd.DataFrame(omega_factor1)
RD=pd.DataFrame(R).T
zuhe=(omega_factor1.dot(RD)).T
print(zuhe)
print(RD)
plt.figure()
plt.plot(date[1:],zuhe.values,lw=3)#随时间变化的组合投资权重
plt.title('Optimal weighted portfolio to minimize variance',fontsize=25)
plt.xlabel('datetime',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('yield',fontsize=20)
plt.show()
#另解
# ddata = np.concatenate((ddata, np.array(omega_factor1).reshape(1, 10)), axis=0)
# ddata = pd.DataFrame(ddata)
# ddata.index = date[1:]
# #print("因子模型估计法：10个行业的最优投资权重的时间序列", ddata)
# plt.figure(1)
# plt.title('最小化期望风险')
# for i in range(10):
#     plt.plot(date[1:], ddata.iloc[:, i])
# plt.show()


#第三题
# #最大化期望效用权重选取压缩估计法计算结果
uhat = np.mean(R, axis=0)
mu = np.mean(uhat)
uhat=uhat.reshape([10,1])#改
uhat=np.mat(uhat)
print('均值',uhat)
gama=3
Q=gama*Cov_Shrink#压缩估计
c=-uhat
b=1
A=np.mat(np.concatenate([np.ones((len(uhat),1))],axis=1)).T
#A=np.mat(np.ones(len(uhat)))
Iden=np.mat(np.identity(n))#n=10返回表示数组。是主对角线是1的方阵，dtype，可选参数，输出数据类型，默认是float型。

omega_factor2=Q.I*A.T*(A*Q.I*A.T).I-Q.I*(Iden-A.T*(A*Q.I*A.T).I*A*Q.I)*c
print('压缩模型估计法在最大化期望效用条件下的权重',omega_factor2)
omega_factor2=omega_factor2.reshape(10)#改
omega_factor2=pd.DataFrame(omega_factor2)
RD=pd.DataFrame(R).T
zuhe2=(omega_factor2.dot(RD)).T

plt.figure()
plt.plot(date[1:],zuhe2.values,lw=3)
plt.title('Optimal weighted portfolio to maximize utility',fontsize=25)
plt.xlabel('datetime',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('yield',fontsize=20)
plt.show()



#错的要乘以A
# w1=np.array([0.04226368,0.04389276,0.15042579,0.12036516,0.10584758,0.10105349,0.06569566,0.13115416,0.1687367 ,0.07056503])#最小化期望方差
# w2=np.array([0.05560063,-0.05324236,-0.03353258, 0.15293537, 0.15196867,0.0676895 , 0.10771742, 0.25929755, 0.27571034,0.01585545])#最大化期望效用
# w1=np.mat(w1)#转为matrix
# w2=np.mat(w2)
# print(w1,w2)
# Rf=np.average(rf)#无风险收益率
# Rw1=R*w1.T
# Rw2=R*w2.T
# Rw1_av=np.average(Rw1)#均值
# Rw1_std=np.std(Rw1)#标准差
# Rw1_sp=(Rw1_av-Rf)/Rw1_std#夏普比率
# Rw2_av=np.average(Rw2)#均值
# Rw2_std=np.std(Rw2)#标准差
# Rw2_sp=(Rw2_av-Rf)/Rw2_std#夏普比率
# print('最小方差的 均值{:.4f},标准差{:.4f},夏普比率{:.4f}'.format(Rw1_av, Rw1_std, Rw1_sp))
# print('最大化期望效用的 均值{:.4f},标准差{:.4f},夏普比率{:.4f}'.format(Rw2_av, Rw2_std, Rw2_sp))
#lg

#print("因子模型估计法：10个行业的最优投资权重的时间序列", ddata)
# ddata = np.zeros((1, 10))
# for i in range(2, len(mkt)):
#     X = np.mat(np.concatenate([np.ones((i, 1)), mkt[0:i, None]], axis=1))
#     Y = np.mat(R[0:i, :])
#     AB_hat = (X.T*X).I*(X.T*Y)
#     ALPHA = AB_hat[0]
#     BETA = AB_hat[1]
#     RESD = Y - X*AB_hat
#     covfactor = np.cov(mkt[1:i + 1])
#     covresidual = np.diag(np.diag(np.cov(RESD, rowvar=False)))
#     Cov_Factor = BETA.T*covfactor*BETA + covresidual
#     uhat = np.mean(R[0:i + 1, :], axis=0)
#     A = np.mat(np.concatenate([uhat[:, None], np.ones((len(uhat), 1))], axis=1)).T
#     up = np.mean(uhat)
#     b = np.mat(np.array([up, 1])[:, None])
#     omega_Factor = Cov_Factor.I*A.T*(A*Cov_Factor.I*A.T).I*b
#     ddata = np.concatenate((ddata, np.array(omega_Factor).reshape(1, 10)), axis=0)
# ddata = pd.DataFrame(ddata)
# ddata.index = d[1:]
# #print("因子模型估计法：10个行业的最优投资权重的时间序列", ddata)
# plt.figure(1)
# plt.title('最小化期望风险')
# for i in range(10):
#     plt.plot(d[1:], ddata.iloc[:, i])
# plt.show()
# print('因子模型估计法的结果为：\n',Cov_Factor)
#另解
#最小期望方差
uhat = np.mean(R, axis=0)
A = np.mat(np.concatenate([uhat[:, None], np.ones((len(uhat), 1))], axis=1)).T
up = np.mean(uhat)
b = np.mat(np.array([up, 1])[:, None])
omega_Factor = Cov_Factor.I*A.T*(A*Cov_Factor.I*A.T).I*b
print('最小期望方差权重',omega_Factor.T)
u_min_uni = np.mean(np.dot(omega_Factor.T,A.T))
d_min_uni = np.std(np.dot(omega_Factor.T,A.T))
xiapu_min_uni = (u_min_uni - np.mean(rf))/ d_min_uni
print('最小期望方差的均值：',u_min_uni)
print('最小期望方差的标准差：',d_min_uni)
print('最小期望方差的夏普比率：',xiapu_min_uni)


#最大化期望效用
Q = 3 * Cov_Shrink
Q_1 = np.linalg.inv(Q)
c = -1 * uhat
c.shape = 10,1#改
b0 = 1
In = np.ones((10,1))#改
S = np.dot(np.dot(In.T,Q_1),In)
S_1 = np.linalg.inv(S)
Q_1AT = np.dot(Q_1,In)
part1 = np.dot(Q_1AT,S_1)
AQ_1 = np.dot(In.T,Q_1)
part2 = np.eye(10) - np.dot(np.dot(In,S_1),AQ_1)#改
max_uni = part1 - np.dot(Q_1,np.dot(part2,c))
print('最大效用权重',max_uni.T)
u_max_uni = np.mean(np.dot(max_uni.T,A.T))
d_max_uni = np.std(np.dot(max_uni.T,A.T))
xiapu_max_uni = (u_max_uni - np.mean(rf))/ d_max_uni
print('最大效用权重的均值：',u_max_uni)
print('最大效用权重的标准差：',d_max_uni)
print('最大效用权重的夏普比率：',xiapu_max_uni)