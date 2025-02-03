import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import pi
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({"font.family":"STIXGeneral","font.size":20,"mathtext.fontset":"cm"})

data1=pd.read_csv('RESSET_IDXMONRET_1.csv',sep=',',encoding='GB2312')
data1.columns=['code','name','date','close','ret']
data2=pd.read_csv('RESSET_THRFACDAT_MONTHLY_1-2.csv',sep=',',encoding='GB2312')
data2.columns=['date','rmrf','smb','hml']
rf=pd.read_csv('RESSET_BDMONRFRET_1.csv',sep=',',encoding='GB2312')

p1=data1[data1['name']=='上证能源']
p2=data1[data1['name']=='上证材料']
p3=data1[data1['name']=='上证工业']
p4=data1[data1['name']=='上证可选']
p5=data1[data1['name']=='上证消费']
p6=data1[data1['name']=='上证医药']
p7=data1[data1['name']=='上证金融']
p8=data1[data1['name']=='上证信息']
p9=data1[data1['name']=='上证电信']
p10=data1[data1['name']=='上证公用']
print(p1)
date=p1['date']
date=pd.to_datetime(date)
p1=p1.iloc[7*12:,3].values
p2=p2.iloc[7*12:,3].values
p3=p3.iloc[7*12:,3].values
p4=p4.iloc[7*12:,3].values
p5=p5.iloc[7*12:,3].values
p6=p6.iloc[7*12:,3].values
p7=p7.iloc[7*12:,3].values
p8=p8.iloc[7*12:,3].values
p9=p9.iloc[7*12:,3].values
p10=p10.iloc[7*12:,3].values
mkt=data2.iloc[7*12:,1].values
smb=data2.iloc[7*12:,2].values
hml=data2.iloc[7*12:,3].values
rf=data2.iloc[7*12:,1].values
r1=np.log(p1[1:])-np.log(p1[:-1])
r2=np.log(p2[1:])-np.log(p2[:-1])
r3=np.log(p3[1:])-np.log(p3[:-1])
r4=np.log(p4[1:])-np.log(p4[:-1])
r5=np.log(p5[1:])-np.log(p5[:-1])
r6=np.log(p6[1:])-np.log(p6[:-1])
r7=np.log(p7[1:])-np.log(p7[:-1])
r8=np.log(p8[1:])-np.log(p8[:-1])
r9=np.log(p9[1:])-np.log(p9[:-1])
r10=np.log(p10[1:])-np.log(p10[:-1])
rexc1=r1-rf[1:]
rexc2=r2-rf[1:]
rexc3=r3-rf[1:]
rexc4=r4-rf[1:]
rexc5=r5-rf[1:]
rexc6=r6-rf[1:]
rexc7=r7-rf[1:]
rexc8=r8-rf[1:]
rexc9=r9-rf[1:]
rexc10=r10-rf[1:]


#第一种：样本模型
R=np.concatenate([rexc1[:,None],
                  rexc2[:,None],
                  rexc3[:,None],
                  rexc4[:,None],
                  rexc5[:,None],
                  rexc6[:, None],
                  rexc7[:, None],
                  rexc8[:, None],
                  rexc9[:, None],
                  rexc10[:, None]],
                 axis=1)
Cov_Sample=np.mat(np.cov(R,rowvar=False))
print('样本模型协方差矩阵为：',np.around(Cov_Sample,4))
#第二种 常量估计法
cov_1=np.cov(R,rowvar=False)
cov_chang=cov_1
sum_cov=0
duijiaoxian=np.zeros(10)
qitayuansu=np.zeros(100)
for i in range(0,10):
    duijiaoxian[i]=cov_1[i][i]
for i in range(0,10):
    for j in range(0,10):
        sum_cov=cov_1[i][j]+sum_cov
sum_cov=sum_cov-np.sum(duijiaoxian)
avr_cov=sum_cov/90
duijiaoxian=np.average(duijiaoxian)
for i in range(0,10):
    for j in range(0,10):
        cov_chang[i][j]=avr_cov
for i in range(0, 5):
    cov_chang[i][i] = duijiaoxian
print('常量估计法的协方差矩阵为',np.around(cov_chang,5))

#第三种：因子模型
X=np.mat(np.concatenate([np.ones((len(mkt)-1,1)),mkt[1:,None],smb[1:,None],hml[1:,None]],axis=1))
Y=np.mat(R)
AB_hat=(X.T*X).I*(X.T*Y)
ALPHA=AB_hat[0]
BETA=AB_hat[1:]
RESD=Y-X*AB_hat
covfactor=np.cov([mkt[1:],smb[1:],hml[1:]])
covresidual=np.diag(np.diag(np.cov(RESD,rowvar=False)))
Cov_Factor=BETA.T*covfactor*BETA+covresidual
print('因子模型的协方差矩阵为',np.around(Cov_Factor,4))

#第四种 压缩估计法
n=10
T=len(R)
id=np.ones([10,10])
Id=pd.DataFrame(id)
tr_cov=np.matrix.trace(Cov_Sample)
tr_cov2=np.matrix.trace(Cov_Sample**2)
pp=((1-2/n)*tr_cov2+tr_cov**2)/((T-2/n)*(tr_cov2-tr_cov**2/n))
c=pp[0,0]
Cov_Shrink=(1-c)*Cov_Sample+c*Cov_Factor
print(c)
print('压缩估计法的协方差矩阵为',np.around(Cov_Shrink,4))

#第五种 指数加权移动平均估计法计算协方差矩阵：2017年6月，2018年8月和2019年11月的计算结果
lamada=0.95
m1=7*12+6
m2=8*12+8
m3=9*12+11
R_pd=pd.DataFrame(R)
R_pd=R_pd.iloc[7*12-1:,:]
rmat0=np.matrix(R_pd)
rmat1=rmat0[5-1:,:]
rmat2=rmat0[12+7-1:,:]
rmat3=rmat0[12*2+10-1:,:]
r0_avr=np.mean(rmat0,axis=0)
r1_avr=np.mean(rmat1,axis=0)
r2_avr=np.mean(rmat1,axis=0)
r3_avr=np.mean(rmat1,axis=0)

EWMA1=np.zeros([10,10])
EWMA2=np.zeros([10,10])
EWME3=np.zeros([10,10])

EWMA1=(1-lamada)*(rmat1-r1_avr).T*(rmat1-r1_avr)+lamada*Cov_Sample
EWMA2=(1-lamada)*(rmat2-r2_avr).T*(rmat2-r2_avr)+lamada*EWMA1
EWMA3=(1-lamada)*(rmat3-r3_avr).T*(rmat3-r3_avr)+lamada*EWMA2
print('2017年6月',np.around(EWMA1,4))
print('2018年8月',np.around(EWMA2,4))
print('2019年11月',np.around(EWMA3,4))

#因子模型估计收益率
data_factors=pd.DataFrame([date,mkt,smb,hml]).T
data_factors.columns=['date','mkt','smb','hml']
data_factors['date']=pd.to_datetime(data_factors['date'])
data_factors['yearmonth']=data_factors['date'].dt.strftime('%Y%m')
data_factors.index=data_factors['date']
print(data_factors)

data_index=pd.DataFrame([date[1:],rexc1,rexc2,rexc3,rexc4,rexc5,rexc6,rexc7,rexc8,rexc9,rexc10]).T
data_index.columns=['date','nengyuan','cailiao','gongye','kexuan','xiaofei','yiyao','jinrong','xinxi','dianxin','gongyong']
data_index['date']=pd.to_datetime(data_index['date'])
data_index['yearmonth']=data_index['date'].dt.strftime('%Y%m')
data_index.index=data_index['date']
print(data_index)

data=pd.merge(left=data_index[['yearmonth','nengyuan','cailiao','gongye','kexuan','xiaofei','yiyao','jinrong','xinxi','dianxin','gongyong']],
                     right=data_factors[['yearmonth','mkt','smb','hml']],
                     on=['yearmonth'],
                     how='inner')
print(data)
data.columns=['yearmonth','nengyuan','cailiao','gongye','kexuan','xiaofei','yiyao','jinrong','xinxi','dianxin','gongyong','mkt','smb','hml']
data.dropna(inplace=True)
data=data.astype('float64')

x=data.loc[:,['mkt','smb','hml']].values

y1=data.loc[:,['nengyuan']].values
y2=data.loc[:,['cailiao']].values
y3=data.loc[:,['gongye']].values
y4=data.loc[:,['kexuan']].values
y5=data.loc[:,['xiaofei']].values
y6=data.loc[:,['yiyao']].values
y7=data.loc[:,['jinrong']].values
y8=data.loc[:,['xinxi']].values
y9=data.loc[:,['dianxin']].values
y10=data.loc[:,['gongyong']].values

#单资产检验
x=sm.add_constant(x)
model1=sm.OLS(y1,x)
model2=sm.OLS(y2,x)
model3=sm.OLS(y3,x)
model4=sm.OLS(y4,x)
model5=sm.OLS(y5,x)
model6=sm.OLS(y6,x)
model7=sm.OLS(y7,x)
model8=sm.OLS(y8,x)
model9=sm.OLS(y9,x)
model10=sm.OLS(y10,x)
results1=model1.fit()
results2=model2.fit()
results3=model3.fit()
results4=model4.fit()
results5=model5.fit()
results6=model6.fit()
results7=model7.fit()
results8=model8.fit()
results9=model9.fit()
results10=model10.fit()
print(results1.summary())
print(results2.summary())
print(results3.summary())
print(results4.summary())
print(results5.summary())
print(results6.summary())
print(results7.summary())
print(results8.summary())
print(results9.summary())
print(results10.summary())
mkt=data['mkt'].values
smb=data['smb'].values
hml=data['hml'].values

#多资产检验

T=len(y1)
N=10#10个资产
K=3#k是因子个数
y=data.iloc[:,1:11].values
xTx=np.dot(np.transpose(x),x)
xTy=np.dot(np.transpose(x),y)
AB_hat=np.dot(np.linalg.inv(xTx),xTy)
ALPHA=AB_hat[0]
print(ALPHA)

RESD=y-np.dot(x,AB_hat)
COV=np.dot(np.transpose(RESD),RESD)/T
invCOV=np.linalg.inv(COV)

fs=x[:,[1,2,3]]
muhat=np.mean(fs,axis=0).reshape((3,1))
fs=fs-np.mean(fs,axis=0)
omegahat=np.dot(np.transpose(fs),fs)/T
invOMG=np.linalg.inv(omegahat)

xxx=np.dot(np.dot(np.transpose(muhat),invOMG),muhat)
yyy=np.dot(np.dot(ALPHA,invCOV),np.transpose(ALPHA))
GRS=(T-N-K)/N*(1/(1+xxx))*yyy

pvalue=1-f.cdf(GRS[0][0],N,T-N-K)
GRS=GRS[0][0]
print(GRS)
print(pvalue)


print('三因子模型的多资产检验结果')
print('{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s},{:>7s}'.format('alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','alpha7','alpha8','alpha9','alpha10','GRS','pvalue'))
print('{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:7.4f}，{:.4f},{:.4f}'.format(ALPHA[0],ALPHA[1],ALPHA[2],ALPHA[3],ALPHA[4],ALPHA[5],ALPHA[6],ALPHA[7],ALPHA[8],ALPHA[9],GRS,pvalue))

def YYZHOU_LSQ(x,SMB,HML,y):
    n=len(SMB)
    sumx=np.sum(x)
    sumSMB=np.sum(SMB)
    sumHML=np.sum(HML)
    sumx2=np.sum(x**2)
    sumSMB2=np.sum(SMB**2)
    sumHML2=np.sum(HML**2)
    sumxSMB= np.sum(x*SMB)
    sumxHML=np.sum(x*HML)
    sumHMLSMB= np.sum(HML*SMB)
    sumy= np.sum(y)
    sumxy=np.sum(x*y)
    sumHMLy= np.sum(HML*y)
    sumSMBy = np.sum(SMB*y)
    A = np.array([[n, sumx, sumSMB, sumHML], [sumx, sumx2, sumxSMB, sumxHML],
                  [sumSMB, sumxSMB, sumSMB2, sumHMLSMB], [sumHML, sumxHML, sumHMLSMB, sumHML2]])
    b = np.array([[sumy], [sumxy], [sumSMBy], [sumHMLy]])
    parLSQ = np.dot(np.linalg.inv(A), b)
    return parLSQ

parLSQ1=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc1)
parLSQ2=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc2)
parLSQ3=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc3)
parLSQ4=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc4)
parLSQ5=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc5)
parLSQ6=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc6)
parLSQ7=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc7)
parLSQ8=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc8)
parLSQ9=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc9)
parLSQ10=YYZHOU_LSQ(mkt[1:],smb[1:],hml[1:],rexc10)
parLSQ1=parLSQ1.reshape(4)
parLSQ2=parLSQ2.reshape(4)
parLSQ3=parLSQ3.reshape(4)
parLSQ4=parLSQ4.reshape(4)
parLSQ5=parLSQ5.reshape(4)
parLSQ6=parLSQ6.reshape(4)
parLSQ7=parLSQ7.reshape(4)
parLSQ8=parLSQ8.reshape(4)
parLSQ9=parLSQ9.reshape(4)
parLSQ10=parLSQ10.reshape(4)
print('上证能源的最小二乘回归系数',parLSQ1)
xishu=pd.DataFrame([parLSQ1,parLSQ2,parLSQ3,parLSQ4,parLSQ5,parLSQ6,parLSQ7,parLSQ8,parLSQ9,parLSQ10])
#xishu=np.mat(xishu.values)
factor=pd.DataFrame([np.ones(len(mkt)-1),mkt[1:],smb[1:],hml[1:]])
mu=xishu.dot(factor).T
#mu.columns=['上证能源','上证材料','上证工业','上证可选','上证消费','上证医药','上证金融','上证信息','上证电信','上证公用']
print(mu)

#最小化期望方差权重
uhat=np.mean(mu,axis=0)
A=np.mat(np.concatenate([uhat[:,None],np.ones((len(uhat),1))],axis=1)).T
up=np.mean(uhat)
b=np.mat(np.array([up,1])[:,None])
omega_factor1=Cov_Factor.I*A.T*(A*Cov_Factor.I*A.T).I*b
print('因子模型估计法在最小化风险条件下的权重',omega_factor1)
omega_factor1=omega_factor1.reshape(10)
omega_factor1=pd.DataFrame(omega_factor1)
RD=pd.DataFrame(R).T
zuhe=(omega_factor1.dot(RD)).T


plt.figure()
plt.plot(date[7*12+1:],zuhe.values,lw=3)
plt.title('Optimal weighted portfolio to minimize variance',fontsize=25)
plt.xlabel('datetime',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('yield',fontsize=20)
plt.show()


#最大化期望效用权重
uhat=np.mean(mu,axis=0)
uhat=uhat.values
uhat=uhat.reshape([10,1])
uhat=np.mat(uhat)
#print('均值',uhat)
gama=3
Q=gama*Cov_Shrink
c=-uhat
b=1
A=np.mat(np.concatenate([np.ones((len(uhat),1))],axis=1)).T
#A=np.mat(np.ones(len(uhat)))
Iden=np.mat(np.identity(n))

omega_factor2=Q.I*A.T*(A*Q.I*A.T).I-Q.I*(Iden-A.T*(A*Q.I*A.T).I*A*Q.I)*c
print('压缩模型估计法在最大化期望效用条件下的权重',omega_factor2)
omega_factor2=omega_factor2.reshape(10)
omega_factor2=pd.DataFrame(omega_factor2)
RD=pd.DataFrame(R).T
zuhe2=(omega_factor2.dot(RD)).T

plt.figure()
plt.plot(date[7*12+1:],zuhe2.values,lw=3)
plt.title('Optimal weighted portfolio to maximize utility',fontsize=25)
plt.xlabel('datetime',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('yield',fontsize=20)
plt.show()
data1=pd.read_csv('RESSET_IDXMONRET_1.csv',sep=',',encoding='GB2312')
data1.columns=['code','name','date','close','ret']
data2=pd.read_csv('RESSET_THRFACDAT_MONTHLY_1-2.csv',sep=',',encoding='GB2312')
data2.columns=['date','rmrf','smb','hml']
rf=pd.read_csv('RESSET_BDMONRFRET_1.csv',sep=',',encoding='GB2312')

p1=data1[data1['name']=='上证能源']
p2=data1[data1['name']=='上证材料']
p3=data1[data1['name']=='上证工业']
p4=data1[data1['name']=='上证可选']
p5=data1[data1['name']=='上证消费']
p6=data1[data1['name']=='上证医药']
p7=data1[data1['name']=='上证金融']
p8=data1[data1['name']=='上证信息']
p9=data1[data1['name']=='上证电信']
p10=data1[data1['name']=='上证公用']

p1=p1.iloc[:7*12:,3].values
p2=p2.iloc[:7*12:,3].values
p3=p3.iloc[:7*12:,3].values
p4=p4.iloc[:7*12:,3].values
p5=p5.iloc[:7*12:,3].values
p6=p6.iloc[:7*12:,3].values
p7=p7.iloc[:7*12:,3].values
p8=p8.iloc[:7*12:,3].values
p9=p9.iloc[:7*12:,3].values
p10=p10.iloc[:7*12:,3].values
mkt=data2.iloc[:7*12:,1].values
smb=data2.iloc[:7*12:,2].values
hml=data2.iloc[:7*12:,3].values
rf=data2.iloc[:7*12:,1].values
r1=np.log(p1[1:])-np.log(p1[:-1])
r2=np.log(p2[1:])-np.log(p2[:-1])
r3=np.log(p3[1:])-np.log(p3[:-1])
r4=np.log(p4[1:])-np.log(p4[:-1])
r5=np.log(p5[1:])-np.log(p5[:-1])
r6=np.log(p6[1:])-np.log(p6[:-1])
r7=np.log(p7[1:])-np.log(p7[:-1])
r8=np.log(p8[1:])-np.log(p8[:-1])
r9=np.log(p9[1:])-np.log(p9[:-1])
r10=np.log(p10[1:])-np.log(p10[:-1])
rexc1=r1-rf[1:]
rexc2=r2-rf[1:]
rexc3=r3-rf[1:]
rexc4=r4-rf[1:]
rexc5=r5-rf[1:]
rexc6=r6-rf[1:]
rexc7=r7-rf[1:]
rexc8=r8-rf[1:]
rexc9=r9-rf[1:]
rexc10=r10-rf[1:]

R=pd.DataFrame([rexc1,rexc2,rexc3,rexc4,rexc5,rexc6,rexc7,rexc8,rexc9,rexc10]).T
R=np.mat(R)
w1=np.array([0.06492814,0.10494351,0.24985955,0.04939334,0.07041483,0.06467587,0.13667438,0.14787028,0.06292788,0.04831222])
w2=np.array([-7.01186161,2.640536,-4.24642146,-4.77861484,5.0664337,0.40613092,7.65070109,-0.27217945,1.1907421,0.35453355])
w1=np.mat(w1)
w2=np.mat(w2)
print(R,w1,w2)
Rf=np.average(rf)
Rw1=R*w1.T
Rw2=R*w2.T
Rw1_av=np.average(Rw1)
Rw1_std=np.std(Rw1)
Rw1_sp=(Rw1_av-Rf)/Rw1_std
Rw2_av=np.average(Rw2)
Rw2_std=np.std(Rw2)
Rw2_sp=(Rw2_av-Rf)/Rw2_std
print(Rw1_av,Rw1_std,Rw2_av,Rw2_std)
print(Rw1_sp,Rw2_sp)



