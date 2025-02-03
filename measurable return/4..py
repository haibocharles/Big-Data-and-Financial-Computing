import pandas as pd
import baostock as bs
import numpy as np
import statsmodels.api as sm
from math import pi
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
import warnings
warnings.filterwarnings('ignore')
#处理变量
data=pd.read_excel('1EData_PredictorData2019.xlsx',sheet_name='Monthly')
data['DP']=data['D12'].apply(np.log)-data['Index'].apply(np.log)
data['EP']=data['E12'].apply(np.log)-data['Index'].apply(np.log)

data['VOL']=data['CRSP_SPvw'].abs().rolling(window=12).mean()*np.sqrt(pi/6)
data['BILL']=data['tbl']-data['tbl'].rolling(window=12).mean()
data['BOND']=data['lty']-data['lty'].rolling(window=12).mean()
data['TERM']=data['lty']-data['tbl']
data['CREDIT']=data['AAA']-data['lty']
data['MA112']=data['Index']>=data['Index'].rolling(window=12).mean()
data['MA312']=data['Index'].rolling(window=3).mean()>=data['Index'].rolling(window=12).mean()
data['MOM6']=data['Index']>=data['Index'].shift(periods=6)
data['PPIG']=data['PPIG']
data['IPG']=data['IPG']

data['ExRet']=data['CRSP_SPvw']-data['Rfree']
data[['MA112','MA312','MOM6']]=data[['MA112','MA312','MOM6']].astype(int)
#ppig ipg
data=pd.concat([data[['yyyymm','CRSP_SPvw','Rfree','ExRet',
                      'DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
                      'MA112','MA312','MOM6']],
                data[['DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
                      'MA112','MA312','MOM6']].shift(periods=1)],axis=1)
print(data)

data.columns=['yyyymm','Ret','Rfree','ExRet',
              'DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
              'MA112','MA312','MOM6',
              'DPL1','EPL1','VOLL1','BILLL1','BONDL1','TERML1','CREDITL1','PPIGL1','IPGL1',
              'MA112L1','MA312L1','MOM6L1']

data=data[data['yyyymm']>=192701]
data.reset_index(drop=True,inplace=True)

data['date']=pd.to_datetime(data['yyyymm'],format='%Y%m')
print(data)
plt.figure(1)
plt.plot(data['date'],data['DP'])
plt.title('DP')
plt.show()
plt.plot(data['date'],data['EP'])
plt.title('EP')
plt.show()
plt.plot(data['date'],data['VOL'])
plt.title('VOL')
plt.show()
plt.plot(data['date'],data['BILL'])
plt.title('BILL')
plt.show()
plt.plot(data['date'],data['BOND'])
plt.title('BOND')
plt.show()
plt.plot(data['date'],data['TERM'])
plt.title('TERM')
plt.show()
plt.plot(data['date'],data['CREDIT'])
plt.title('CREDIT')
plt.show()
plt.plot(data['date'],data['PPIG'])
plt.title('PPIG')
plt.show()
plt.plot(data['date'],data['IPG'])
plt.title('IPG')
plt.show()
plt.plot(data['date'],data['MA112'])
plt.title('MA112')
plt.show()
plt.plot(data['date'],data['MA312'])
plt.title('MA312')
plt.show()
plt.plot(data['date'],data['MOM6'])
plt.title('MOM6')
plt.show()
print(data['yyyymm'])
# 单因子模型(双变量预测模型)：
def myfun_stat_gains(rout,rmean,rreal):
    R2os=1-np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)
    d=(rreal-rmean)**2-((rreal-rout)**2-(rmean-rout)**2)
    x=sm.add_constant(np.arange(len(d))+1)
    model=sm.OLS(d,x)
    # MSFE-adjusted 统计量：回归di和i，常数项的t统计量
    fitres=model.fit()
    MFSRadj=fitres.tvalues[0]
    pvalue_MFSEadj=fitres.pvalues[0]
    if(R2os>0)&(pvalue_MFSEadj<=0.01):
        jud='在1%的显著性水平下有样本外预测能力'
    elif(R2os>0)&(pvalue_MFSEadj>0.01)&(pvalue_MFSEadj<=0.05):
        jud = '在5%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MFSEadj > 0.05) & (pvalue_MFSEadj <= 0.1):
        jud = '在10%的显著性水平下有样本外预测能力'
    else:
        jud='无样本外预测能力'
    print('Stat gains:R2os={:f},MFSEadj={:f},MFSEpvalue={:f}'.format(R2os,MFSRadj,pvalue_MFSEadj))
    print('Inference:{:s}'.format(jud))

    return R2os,MFSRadj,pvalue_MFSEadj

def myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5):
    # Out-of-sample tests 经济显著性检验
    omg_out=rout/volt2/gmm#风险系数=5
    rp_out=rfree+omg_out*rreal
    Uout=np.mean(rp_out)-0.5*gmm*np.var(rp_out)
    omg_mean=rmean/volt2/gmm
    rp_mean=rfree+omg_mean*rreal
    Umean=np.mean(rp_mean)-0.5*gmm*np.var(rp_mean)
    DeltaU=Uout-Umean
    # 检验𝒓 ̂的均值不为0，计算utility gain=Uout-Umean
    if DeltaU<10**-6:
        jud='没有经济意义'
    else:
        jud='有经济意义'
    print('Econ Gains:Delta U={:f},Umean={:f}'.format(DeltaU,Uout,Umean))
    print('Inference:{:s}'.format(jud))

    return Uout,Umean,DeltaU
# 因子构建预测效力的指标
#样本内检验
#单因子模型：OLS线性拟合
factor='DP'
model=smf.ols('ExRet~DPL1',data=data[['ExRet','DPL1']])
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['DPL1']
rg_DP_pvalue=results.pvalues['DPL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='EP'
model=smf.ols('ExRet~EPL1',data=data[['ExRet','EPL1']])
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['EPL1']
rg_DP_pvalue=results.pvalues['EPL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='VOL'
model=smf.ols('ExRet~VOLL1',data=data[['ExRet','VOLL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['VOLL1']
rg_DP_pvalue=results.pvalues['VOLL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='BILL'
model=smf.ols('ExRet~BILLL1',data=data[['ExRet','BILLL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['BILLL1']
rg_DP_pvalue=results.pvalues['BILLL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='BOND'
model=smf.ols('ExRet~BONDL1',data=data[['ExRet','BONDL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['BONDL1']
rg_DP_pvalue=results.pvalues['BONDL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='TERM'
model=smf.ols('ExRet~TERML1',data=data[['ExRet','TERML1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['TERML1']
rg_DP_pvalue=results.pvalues['TERML1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='CREDIT'
model=smf.ols('ExRet~CREDITL1',data=data[['ExRet','CREDITL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['CREDITL1']
rg_DP_pvalue=results.pvalues['CREDITL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='MA112'
model=smf.ols('ExRet~MA112L1',data=data[['ExRet','MA112L1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['MA112L1']
rg_DP_pvalue=results.pvalues['MA112L1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='MA312'
model=smf.ols('ExRet~MA312L1',data=data[['ExRet','MA312L1']])
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['MA312L1']
rg_DP_pvalue=results.pvalues['MA312L1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='MOM6'
model=smf.ols('ExRet~MOM6L1',data=data[['ExRet','MOM6L1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['MOM6L1']
rg_DP_pvalue=results.pvalues['MOM6L1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='PPIG'
model=smf.ols('ExRet~PPIGL1',data=data[['ExRet','PPIGL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['PPIGL1']
rg_DP_pvalue=results.pvalues['PPIGL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='IPG'
model=smf.ols('ExRet~IPGL1',data=data[['ExRet','IPGL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['IPGL1']
rg_DP_pvalue=results.pvalues['IPGL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))
#样本外检验
#单因子模型：OLS线性拟合
factor_out='DP'
datafit=data[['yyyymm','Ret','Rfree','ExRet','DP','DPL1']].copy(deep=True)#年份,收益率,无风险超额,DP,DPLOG
n_in=np.sum(datafit['yyyymm']<=195612)#样本内
n_out=np.sum(datafit['yyyymm']>195612)#样本外
rout=np.zeros(n_out)#预测的坑
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)#波动率
for i in range(n_out):#预测nout次
    model=smf.ols('ExRet~DPL1',data=datafit[['ExRet','DPL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['DPL1']
    f=datafit['DP'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='EP'
datafit=data[['yyyymm','Ret','Rfree','ExRet','EP','EPL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~EPL1',data=datafit[['ExRet','EPL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['EPL1']
    f=datafit['EP'].iloc[n_in+i-1]#前一天的因子
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)#最近12个月的平方和
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='VOL'
datafit=data[['yyyymm','Ret','Rfree','ExRet','VOL','VOLL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~VOLL1',data=datafit[['ExRet','VOLL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['VOLL1']
    f=datafit['VOL'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='BILL'
datafit=data[['yyyymm','Ret','Rfree','ExRet','BILL','BILLL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~BILLL1',data=datafit[['ExRet','BILLL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['BILLL1']
    f=datafit['BILL'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='BOND'
datafit=data[['yyyymm','Ret','Rfree','ExRet','BOND','BONDL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~BONDL1',data=datafit[['ExRet','BONDL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['BONDL1']
    f=datafit['BOND'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='TERM'
datafit=data[['yyyymm','Ret','Rfree','ExRet','TERM','TERML1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~TERML1',data=datafit[['ExRet','TERML1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['TERML1']
    f=datafit['TERM'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='CREDIT'
datafit=data[['yyyymm','Ret','Rfree','ExRet','CREDIT','CREDITL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~CREDITL1',data=datafit[['ExRet','CREDITL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['CREDITL1']
    f=datafit['CREDIT'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='MA112'
datafit=data[['yyyymm','Ret','Rfree','ExRet','MA112','MA112L1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~MA112L1',data=datafit[['ExRet','MA112L1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['MA112L1']
    f=datafit['MA112'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='MA312'
datafit=data[['yyyymm','Ret','Rfree','ExRet','MA312','MA312L1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~MA312L1',data=datafit[['ExRet','MA312L1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['MA312L1']
    f=datafit['MA312'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='MOM6'
datafit=data[['yyyymm','Ret','Rfree','ExRet','MOM6','MOM6L1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~MOM6L1',data=datafit[['ExRet','MOM6L1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['MOM6L1']
    f=datafit['MOM6'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='PPIG'
datafit=data[['yyyymm','Ret','Rfree','ExRet','PPIG','PPIGL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~PPIGL1',data=datafit[['ExRet','PPIGL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['PPIGL1']
    f=datafit['PPIG'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='IPG'
datafit=data[['yyyymm','Ret','Rfree','ExRet','IPG','IPGL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~IPGL1',data=datafit[['ExRet','IPGL1']].iloc[:(n_in+i),:])
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['IPGL1']
    f=datafit['IPG'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

#样本外检验
#多因子模型：OLS线性拟合
factor_out='DP,EP,VOL,BILL,BOND,TERM,CREDIT,PPIG,IPG,MA112,MA312,MOM6'
datafit=data.copy(deep=True)

n_in=np.sum(datafit['yyyymm']<=195612)
n_out=np.sum(datafit['yyyymm']>195612)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)

for i in range(n_out):
    model=smf.ols('ExRet~DPL1+EPL1+VOLL1+BILLL1+BONDL1+TERML1+CREDITL1+'
                  'PPIGL1+IPGL1+MA112L1+MA312L1+MOM6L1',
                  data=datafit[['ExRet','DPL1','EPL1','VOLL1','BILLL1','BONDL1','TERML1',
                                'CREDITL1','PPIGL1','IPGL1','MA112L1','MA312L1','MOM6L1']].iloc[:(n_in+i),:])
    results=model.fit()
    k=results.params.values
    f=datafit[['DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG',
               'IPG','MA112','MA312','MOM6']].iloc[n_in+i-1,:].values
    f=np.concatenate((np.array([1]),f))
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i]=datafit['Rfree'].iloc[n_in+i]
    rout[i]=np.sum(k*f)
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit



#样本外检验
#多因子模型：LASSO回归，Ridge回归，ElasticNet回归
factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 195612)
n_out = np.sum(datafit['yyyymm'] > 195612)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
#Ridge
reg = sklm.RidgeCV(cv=10, fit_intercept=True, normalize=True)
for i in range(n_out):
    X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
                 'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
                 'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

#Lasso
factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 195612)
n_out = np.sum(datafit['yyyymm'] > 195612)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
reg = sklm.LassoCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
# reg_lasso = linear_model.LassoLarsCV(cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10000000)
# reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
for i in range(n_out):
    X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
                 'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
                 'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

#ElasticNet
factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 195612)
n_out = np.sum(datafit['yyyymm'] > 195612)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
for i in range(n_out):
    X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
                 'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
                 'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

daily_data1=pd.read_csv('RESSET_DRESSTK_1990_2000_1.csv',encoding='GB2312')
daily_data2=pd.read_csv('RESSET_DRESSTK_2001_2010_1-2.csv',encoding='GB2312')
daily_data3=pd.read_csv('RESSET_DRESSTK_2011_2015_1-2.csv',encoding='GB2312')
daily_data4=pd.read_csv('RESSET_DRESSTK_2016_2020_1-2.csv',encoding='GB2312')
daily_data5=pd.read_csv('RESSET_DRESSTK_2021__1-2.csv',encoding='GB2312')


daily_data=daily_data1.append(daily_data2,ignore_index=True)
daily_data=daily_data.append(daily_data3,ignore_index=True)
daily_data=daily_data.append(daily_data4,ignore_index=True)
daily_data=daily_data.append(daily_data5,ignore_index=True)
daily_data['Date']=pd.to_datetime(daily_data['Date'])
daily_data['yearmonth']=daily_data['Date'].dt.strftime('%Y%m')
print(daily_data)

pe_data=pd.read_csv('RESSET_PERATIO_1990_2000_1.csv',encoding='GB2312')
beta_data=pd.read_csv('RESSET_SMONRETBETA_BFDT_1.csv',encoding='GB2312')
month_data=pd.read_csv('RESSET_MRESSTK_1.csv',encoding='GB2312')

month_data=pd.merge(left=month_data[['Stkcd','Date','yyyymm','Trdsum','MonTrdTurnR','Monret','Monrfret','EPS','ROE','IncomePS']],
                     right=beta_data[['Stkcd','Date','Beta']],
                     on=['Stkcd','Date'],
                     how='inner')
month_data=pd.merge(left=month_data,
                     right=pe_data[['Stkcd','Date','PeRatio']],
                     on=['Stkcd','Date'],
                     how='inner')
month_data['Date']=pd.to_datetime(month_data['Date'])
month_data['yearmonth']=month_data['Date'].dt.strftime('%Y%m')
daily_data1=daily_data.copy()
daily_data1.fillna(0)
daily_data.dropna(inplace=True)
M=np.unique(month_data['yearmonth'].values)
vol=np.zeros(len(M))
high=np.zeros(len(M))
for i in range(len(M)):
    a=daily_data[daily_data['yearmonth']==M[i]]
    vol[i]=np.sum(a.iloc[:,-3].values**2)
    high[i]=max(a.iloc[:,2].values)
#vol['yearmonth']=vol['date'].dt.strftime('%Y%m').astype(int)
#date_dt=datetime.datetime.strptime(M,'%Y-%m')
date_dt= pd.date_range('1/2000','1/2022',freq='M')
M=pd.DataFrame(M)
M.columns=['datetime']
date_datetime=date_dt.strftime('%Y%m')
date_dt=pd.DataFrame([date_dt,date_datetime])
date_dt=date_dt.T
date_dt.columns=['date','datetime']
date_dt=pd.merge(left=date_dt,
                     right=M,
                     on=['datetime'],
                     how='inner')
date_dt=date_dt.iloc[:,0]

daily_data['yearmonth']=daily_data['Date'].dt.strftime('%Y%m')
vol_data=pd.DataFrame(vol,index=date_dt,columns=['vol'])
vol_data['Date']=vol_data.index
vol_data['yearmonth']=vol_data['Date'].dt.strftime('%Y%m')
high_data=pd.DataFrame(high,index=date_dt,columns=['high'])
high_data['Date']=high_data.index
high_data['yearmonth']=high_data['Date'].dt.strftime('%Y%m')
month_data=pd.merge(left=month_data,
                     right=vol_data[['yearmonth','vol']],
                     on=['yearmonth'],
                     how='inner')
month_data=pd.merge(left=month_data,
                     right=high_data[['yearmonth','high']],
                     on=['yearmonth'],
                     how='inner')
month_data['liqulity']=month_data['Monret']/np.log(month_data['Trdsum'])
month_data.dropna(inplace=True)
high_point=month_data['high'].values
high=month_data['high'].values
for i in range(3,137):
    high_point[i]=(high[i]/max(high[i-1],high[i-2],high[i-3]))

high_point[0]=11.88/10.86
high_point[1]=11.9/11.88
high_point[2]=11.39/11.9
month_data['high_point']=high_point

lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
rs = bs.query_history_k_data_plus("sh.600855",
                                  "date, time, code, close",
                                  start_date='2000-01-01', end_date='2021-12-31',
                                  frequency="5", adjustflag="3")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
data_list = []

while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

result['close'] = result['close'].apply(float)
close=result.iloc[:,3]
result['ret']=np.log(close)-np.log(close.shift(1))
ret=result['ret']
result.dropna(inplace=True)
print(result)
result['date']=pd.to_datetime(result['date'])
result['yearmonth']=result['date'].dt.strftime('%Y%m')
result['yearmonth']=result['yearmonth'].apply(int)
D= pd.date_range('1/1/2000','1/1/2022',freq='D')
D=np.unique(result['date'].values)
print(D)
rdvar=np.zeros(len(D))
rdskew=np.zeros(len(D))
for i in range(len(D)):
    a=result[result['date']==D[i]]
    rdvar[i]=np.sum(a.iloc[:,-1].values**2)
    N=len(a.iloc[:,-1].values)
    rdskew[i]=np.sqrt(N)*np.sum(a.iloc[:,-1].values**3)/rdvar[i]
rdskew=pd.DataFrame(rdskew)
rdskew['date']=D
rdskew['yearmonth']=rdskew['date'].dt.strftime('%Y%m').astype(int)
rdskew.columns=['rdskew','date','yearmonth']
rdskew.dropna(inplace=True)
print(rdskew)

M=np.unique(rdskew['yearmonth'].values)

rmskew=np.zeros(len(M))
for i in range(len(M)):
    a=rdskew[rdskew['yearmonth']==M[i]]
    N=len(a.iloc[:,-1].values)
    rmskew[i]=np.sum(a.iloc[:,0].values)/N

rmskew=pd.DataFrame(rmskew)
rmskew['yearmonth']=M
rmskew.columns=['rskew','yearmonth']
month_data['yearmonth']=month_data['yearmonth'].astype(int)
month_data=pd.merge(left=month_data,
                     right=rmskew[['rskew','yearmonth']],
                     on=['yearmonth'],
                     how='inner')

month_data['ExRet']=month_data['Monret']-month_data['Monrfret']
month_data.columns=[['Stkcd','date','yyyymm','Trdsum','Turnover','Ret','Rfree','EPS','ROE',
                'IncomePS','Beta','PE','datetime','VOL','h','liqulity','high','rskew','ExRet',]]
print(month_data)
data=pd.concat([month_data[['yyyymm','Ret','Rfree','ExRet',
                                  'PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']],
                month_data[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].shift(periods=1)],axis=1)
data.columns=['yyyymm','Ret','Rfree','ExRet',
                    'PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew',
                    'PEL1','TurnoverL1','VOLL1','EPSL1','ROEL1','IncomePSL1','BetaL1','liqulityL1','highL1','rskewL1']
data['date']=pd.to_datetime(data['yyyymm'],format='%Y%m')
data.reset_index(drop=True,inplace=True)
data.dropna(inplace=True)
print(data)
data.to_excel('datajack.xlsx')
def myfun_stat_gains(rout,rmean,rreal):
    R2os=1-np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)
    d=(rreal-rmean)**2-((rreal-rout)**2-(rmean-rout)**2)
    x=sm.add_constant(np.arange(len(d))+1)
    model=sm.OLS(d,x)
    fitres=model.fit()
    MFSRadj=fitres.tvalues[0]
    pvalue_MFSEadj=fitres.pvalues[0]

    if(R2os>0)&(pvalue_MFSEadj<=0.01):
        jud='在1%的显著性水平下有样本外预测能力'
    elif(R2os>0)&(pvalue_MFSEadj>0.01)&(pvalue_MFSEadj<=0.05):
        jud = '在5%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MFSEadj > 0.05) & (pvalue_MFSEadj <= 0.1):
        jud = '在10%的显著性水平下有样本外预测能力'
    else:
        jud='无样本外预测能力'
    print('Stat gains:R2os={:f},MFSEadj={:f},MFSEpvalue={:f}'.format(R2os,MFSRadj,pvalue_MFSEadj))
    print('Inference:{:s}'.format(jud))

    return R2os,MFSRadj,pvalue_MFSEadj

def myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5):
    omg_out=rout/volt2/gmm
    rp_out=rfree+omg_out*rreal
    Uout=np.mean(rp_out)-0.5*gmm*np.var(rp_out)
    omg_mean=rmean/volt2/gmm
    rp_mean=rfree+omg_mean*rreal
    Umean=np.mean(rp_mean)-0.5*gmm*np.var(rp_mean)
    DeltaU=Uout-Umean

    if DeltaU<10**-6:
        jud='没有经济意义'
    else:
        jud='有经济意义'
    print('Econ Gains:Delta U={:f},Umean={:f}'.format(DeltaU,Uout,Umean))
    print('Inference:{:s}'.format(jud))

    return Uout,Umean,DeltaU

#样本内检验
#单因子模型：OLS线性拟合
factor='PE'
model=smf.ols('ExRet~PEL1',data=data[['ExRet','PEL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['PEL1']
rg_DP_pvalue=results.pvalues['PEL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='Turnover'
model=smf.ols('ExRet~TurnoverL1',data=data[['ExRet','TurnoverL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['TurnoverL1']
rg_DP_pvalue=results.pvalues['TurnoverL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='VOL'
model=smf.ols('ExRet~VOLL1',data=data[['ExRet','VOLL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['VOLL1']
rg_DP_pvalue=results.pvalues['VOLL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='EPS'
model=smf.ols('ExRet~EPSL1',data=data[['ExRet','EPSL1']])
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['EPSL1']
rg_DP_pvalue=results.pvalues['EPSL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='ROE'
model=smf.ols('ExRet~ROEL1',data=data[['ExRet','ROEL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['ROEL1']
rg_DP_pvalue=results.pvalues['ROEL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='IncomePS'
model=smf.ols('ExRet~IncomePSL1',data=data[['ExRet','IncomePSL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['IncomePSL1']
rg_DP_pvalue=results.pvalues['IncomePSL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='Beta'
model=smf.ols('ExRet~BetaL1',data=data[['ExRet','BetaL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['BetaL1']
rg_DP_pvalue=results.pvalues['BetaL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='liqulity'
model=smf.ols('ExRet~liqulityL1',data=data[['ExRet','liqulityL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['liqulityL1']
rg_DP_pvalue=results.pvalues['liqulityL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='high'
model=smf.ols('ExRet~highL1',data=data[['ExRet','highL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['highL1']
rg_DP_pvalue=results.pvalues['highL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='rskew'
model=smf.ols('ExRet~rskewL1',data=data[['ExRet','rskewL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['rskewL1']
rg_DP_pvalue=results.pvalues['rskewL1']
if rg_DP_pvalue<=0.01:
    jud='在1%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
    jud='在5%的显著性水平下有样本内预测能力'
elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
    jud = '在10%的显著性水平下有样本内预测能力'
else:
    jud='无样本内预测能力'
print('In-sample tests for one factor model with OLs:')
print('Predictor:{:s}'.format(factor))
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))
#样本外检验
#单因子模型：OLS线性拟合
print(data['yyyymm'])
factor_out='PE'
datafit=data[['yyyymm','Ret','Rfree','ExRet','PE','PEL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~PEL1',data=datafit[['ExRet','PEL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['PEL1']
    f=datafit['PE'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='Turnover'
datafit=data[['yyyymm','Ret','Rfree','ExRet','Turnover','TurnoverL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~TurnoverL1',data=datafit[['ExRet','TurnoverL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['TurnoverL1']
    f=datafit['Turnover'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='VOL'
datafit=data[['yyyymm','Ret','Rfree','ExRet','VOL','VOLL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~VOLL1',data=datafit[['ExRet','VOLL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['VOLL1']
    f=datafit['VOL'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='EPS'
datafit=data[['yyyymm','Ret','Rfree','ExRet','EPS','EPSL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~EPSL1',data=datafit[['ExRet','EPSL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['EPSL1']
    f=datafit['EPS'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='ROE'
datafit=data[['yyyymm','Ret','Rfree','ExRet','ROE','ROEL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~ROEL1',data=datafit[['ExRet','ROEL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['ROEL1']
    f=datafit['ROE'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='IncomePS'
datafit=data[['yyyymm','Ret','Rfree','ExRet','IncomePS','IncomePSL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~IncomePSL1',data=datafit[['ExRet','IncomePSL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['IncomePSL1']
    f=datafit['IncomePS'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='Beta'
datafit=data[['yyyymm','Ret','Rfree','ExRet','Beta','BetaL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~BetaL1',data=datafit[['ExRet','BetaL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['BetaL1']
    f=datafit['Beta'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='liqulity'
datafit=data[['yyyymm','Ret','Rfree','ExRet','liqulity','liqulityL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~liqulityL1',data=datafit[['ExRet','liqulityL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['liqulityL1']
    f=datafit['liqulity'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='high'
datafit=data[['yyyymm','Ret','Rfree','ExRet','high','highL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~highL1',data=datafit[['ExRet','highL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['highL1']
    f=datafit['high'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

factor_out='rskew'
datafit=data[['yyyymm','Ret','Rfree','ExRet','rskew','rskewL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~rskewL1',data=datafit[['ExRet','rskewL1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['rskewL1']
    f=datafit['rskew'].iloc[n_in+i-1]
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in + i]
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit

#样本外检验
#多因子模型：OLS线性拟合
factor_out='PE,Turnover,VOL,EPS,ROE,IncomePS,Beta,liqulity,high,rskew'
datafit=data.copy(deep=True)

n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)

for i in range(n_out):
    model=smf.ols('ExRet~PEL1+TurnoverL1+VOLL1+EPSL1+ROEL1+IncomePSL1+BetaL1+liqulityL1+highL1+rskewL1',
                  data=datafit[['ExRet','PEL1','TurnoverL1','VOLL1','EPSL1','ROEL1','IncomePSL1','BetaL1','liqulityL1','highL1','rskewL1']].iloc[:(n_in+i),:])
    results=model.fit()
    k=results.params.values
    f=datafit[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].iloc[n_in+i-1,:].values
    f=np.concatenate((np.array([1]),f))
    rreal[i]=datafit['ExRet'].iloc[n_in+i]
    rfree[i]=datafit['Rfree'].iloc[n_in+i]
    rout[i]=np.sum(k*f)
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit


#样本外检验
#多因子模型：LASSO回归，Ridge回归，ElasticNet回归
factor_out='PE,Turnover,VOL,EPS,ROE,IncomePS,Beta,liqulity,high,rskew'
factor_list = np.array(['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 201012)
n_out = np.sum(datafit['yyyymm'] > 201012)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
#Ridge
reg = sklm.RidgeCV(cv=10, fit_intercept=True, normalize=True)
for i in range(n_out):
    X = datafit[['PEL1','TurnoverL1','VOLL1','EPSL1','ROEL1','IncomePSL1','BetaL1','liqulityL1','highL1','rskewL1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

#Lasso
factor_out='PE,Turnover,VOL,EPS,ROE,IncomePS,Beta,liqulity,high,rskew'
factor_list = np.array(['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 201012)
n_out = np.sum(datafit['yyyymm'] > 201012)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
reg = sklm.LassoCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
# reg_lasso = linear_model.LassoLarsCV(cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10000000)
# reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
for i in range(n_out):
    X = datafit[['PEL1','TurnoverL1','VOLL1','EPSL1','ROEL1','IncomePSL1','BetaL1','liqulityL1','highL1','rskewL1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

#ElasticNet
factor_out='PE,Turnover,VOL,EPS,ROE,IncomePS,Beta,liqulity,high,rskew'
factor_list = np.array(['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew'])

datafit = data.copy(deep=True)

n_in = np.sum(datafit['yyyymm'] <= 201012)
n_out = np.sum(datafit['yyyymm'] > 201012)
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)
reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)

for i in range(n_out):
    X = datafit[['PEL1','TurnoverL1','VOLL1','EPSL1','ROEL1','IncomePSL1','BetaL1','liqulityL1','highL1','rskewL1']].iloc[:(n_in+i), :].values
    y = datafit['ExRet'].iloc[:(n_in+i)].values
    reg.fit(X, y)
    # print(factor_list[np.abs(reg.coef_) != 0])
    k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    f = datafit[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].iloc[n_in+i-1, :].values
    f = np.concatenate((np.array([1]), f))
    rreal[i] = datafit['ExRet'].iloc[n_in+i]
    rfree[i] = datafit['Rfree'].iloc[n_in+i]
    rout[i] = np.sum(k*f)
    rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

print()
print('Out-of-sample tests for multi-factor model with ML method:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit

