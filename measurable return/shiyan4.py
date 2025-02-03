import numpy as np
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
# data=pd.read_excel('1EData_PredictorData2019.xlsx',sheet_name='Monthly')
# data['DP']=data['D12'].apply(np.log)-data['Index'].apply(np.log)
# data['EP']=data['E12'].apply(np.log)-data['Index'].apply(np.log)
#
# data['VOL']=data['CRSP_SPvw'].abs().rolling(window=12).mean()*np.sqrt(pi/6)
# data['BILL']=data['tbl']-data['tbl'].rolling(window=12).mean()
# data['BOND']=data['lty']-data['lty'].rolling(window=12).mean()
# data['TERM']=data['lty']-data['tbl']
# data['CREDIT']=data['AAA']-data['lty']
# data['MA112']=data['Index']>=data['Index'].rolling(window=12).mean()
# data['MA312']=data['Index'].rolling(window=3).mean()>=data['Index'].rolling(window=12).mean()
# data['MOM6']=data['Index']>=data['Index'].shift(periods=6)
# data['PPIG']=data['PPIG']
# data['IPG']=data['IPG']
# #
# data['ExRet']=data['CRSP_SPvw']-data['Rfree']
# data[['MA112','MA312','MOM6']]=data[['MA112','MA312','MOM6']].astype(int)
# #ppig ipg
# data=pd.concat([data[['yyyymm','CRSP_SPvw','Rfree','ExRet',
#                       'DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
#                       'MA112','MA312','MOM6']],
#                 data[['DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
#                       'MA112','MA312','MOM6']].shift(periods=1)],axis=1)
# print(data)
#
# data.columns=['yyyymm','Ret','Rfree','ExRet',
#               'DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG',
#               'MA112','MA312','MOM6',
#               'DPL1','EPL1','VOLL1','BILLL1','BONDL1','TERML1','CREDITL1','PPIGL1','IPGL1',
#               'MA112L1','MA312L1','MOM6L1']
#
# data=data[data['yyyymm']>=192701]
# data.reset_index(drop=True,inplace=True)
#
# data['date']=pd.to_datetime(data['yyyymm'],format='%Y%m')
# print(data)
# plt.figure(1)
# plt.plot(data['date'],data['DP'])
# plt.title('DP')
# plt.show()
# plt.plot(data['date'],data['EP'])
# plt.title('EP')
# plt.show()
# plt.plot(data['date'],data['VOL'])
# plt.title('VOL')
# plt.show()
# plt.plot(data['date'],data['BILL'])
# plt.title('BILL')
# plt.show()
# plt.plot(data['date'],data['BOND'])
# plt.title('BOND')
# plt.show()
# plt.plot(data['date'],data['TERM'])
# plt.title('TERM')
# plt.show()
# plt.plot(data['date'],data['CREDIT'])
# plt.title('CREDIT')
# plt.show()
# plt.plot(data['date'],data['PPIG'])
# plt.title('PPIG')
# plt.show()
# plt.plot(data['date'],data['IPG'])
# plt.title('IPG')
# plt.show()
# plt.plot(data['date'],data['MA112'])
# plt.title('MA112')
# plt.show()
# plt.plot(data['date'],data['MA312'])
# plt.title('MA312')
# plt.show()
# plt.plot(data['date'],data['MOM6'])
# plt.title('MOM6')
# plt.show()
# # 单因子模型(双变量预测模型)：
# def myfun_stat_gains(rout,rmean,rreal):
#     R2os=1-np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)
#     d=(rreal-rmean)**2-((rreal-rout)**2-(rmean-rout)**2)
#     x=sm.add_constant(np.arange(len(d))+1)
#     model=sm.OLS(d,x)
#     # MSFE-adjusted 统计量：回归di和i，常数项的t统计量
#     fitres=model.fit()
#     MFSRadj=fitres.tvalues[0]
#     pvalue_MFSEadj=fitres.pvalues[0]
#     if(R2os>0)&(pvalue_MFSEadj<=0.01):
#         jud='在1%的显著性水平下有样本外预测能力'
#     elif(R2os>0)&(pvalue_MFSEadj>0.01)&(pvalue_MFSEadj<=0.05):
#         jud = '在5%的显著性水平下有样本外预测能力'
#     elif (R2os > 0) & (pvalue_MFSEadj > 0.05) & (pvalue_MFSEadj <= 0.1):
#         jud = '在10%的显著性水平下有样本外预测能力'
#     else:
#         jud='无样本外预测能力'
#     print('Stat gains:R2os={:f},MFSEadj={:f},MFSEpvalue={:f}'.format(R2os,MFSRadj,pvalue_MFSEadj))
#     print('Inference:{:s}'.format(jud))
#
#     return R2os,MFSRadj,pvalue_MFSEadj
#
# def myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5):
#     # Out-of-sample tests 经济显著性检验
#     omg_out=rout/volt2/gmm#风险系数=5
#     rp_out=rfree+omg_out*rreal
#     Uout=np.mean(rp_out)-0.5*gmm*np.var(rp_out)
#     omg_mean=rmean/volt2/gmm
#     rp_mean=rfree+omg_mean*rreal
#     Umean=np.mean(rp_mean)-0.5*gmm*np.var(rp_mean)
#     DeltaU=Uout-Umean
#     # 检验𝒓 ̂的均值不为0，计算utility gain=Uout-Umean
#     if DeltaU<10**-6:
#         jud='没有经济意义'
#     else:
#         jud='有经济意义'
#     print('Econ Gains:Delta U={:f},Umean={:f}'.format(DeltaU,Uout,Umean))
#     print('Inference:{:s}'.format(jud))
#
#     return Uout,Umean,DeltaU
# # 因子构建预测效力的指标
# #样本内检验
# #单因子模型：OLS线性拟合
# factor='DP'
# model=smf.ols('ExRet~DPL1',data=data[['ExRet','DPL1']])
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['DPL1']
# rg_DP_pvalue=results.pvalues['DPL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='EP'
# model=smf.ols('ExRet~EPL1',data=data[['ExRet','EPL1']])
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['EPL1']
# rg_DP_pvalue=results.pvalues['EPL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='VOL'
# model=smf.ols('ExRet~VOLL1',data=data[['ExRet','VOLL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['VOLL1']
# rg_DP_pvalue=results.pvalues['VOLL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='BILL'
# model=smf.ols('ExRet~BILLL1',data=data[['ExRet','BILLL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['BILLL1']
# rg_DP_pvalue=results.pvalues['BILLL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='BOND'
# model=smf.ols('ExRet~BONDL1',data=data[['ExRet','BONDL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['BONDL1']
# rg_DP_pvalue=results.pvalues['BONDL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='TERM'
# model=smf.ols('ExRet~TERML1',data=data[['ExRet','TERML1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['TERML1']
# rg_DP_pvalue=results.pvalues['TERML1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='CREDIT'
# model=smf.ols('ExRet~CREDITL1',data=data[['ExRet','CREDITL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['CREDITL1']
# rg_DP_pvalue=results.pvalues['CREDITL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='MA112'
# model=smf.ols('ExRet~MA112L1',data=data[['ExRet','MA112L1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['MA112L1']
# rg_DP_pvalue=results.pvalues['MA112L1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='MA312'
# model=smf.ols('ExRet~MA312L1',data=data[['ExRet','MA312L1']])
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['MA312L1']
# rg_DP_pvalue=results.pvalues['MA312L1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='MOM6'
# model=smf.ols('ExRet~MOM6L1',data=data[['ExRet','MOM6L1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['MOM6L1']
# rg_DP_pvalue=results.pvalues['MOM6L1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='PPIG'
# model=smf.ols('ExRet~PPIGL1',data=data[['ExRet','PPIGL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['PPIGL1']
# rg_DP_pvalue=results.pvalues['PPIGL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
#
# factor='IPG'
# model=smf.ols('ExRet~IPGL1',data=data[['ExRet','IPGL1']])#可以指定回归模型是什么样子
# results=model.fit()
# rg_con=results.params['Intercept']
# rg_con_pvalue=results.pvalues['Intercept']
# rg_DP=results.params['IPGL1']
# rg_DP_pvalue=results.pvalues['IPGL1']
# if rg_DP_pvalue<=0.01:
#     jud='在1%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue>0.01)&(rg_DP_pvalue<=0.05):
#     jud='在5%的显著性水平下有样本内预测能力'
# elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.1):
#     jud = '在10%的显著性水平下有样本内预测能力'
# else:
#     jud='无样本内预测能力'
# print('In-sample tests for one factor model with OLs:')
# print('Predictor:{:s}'.format(factor))
# print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))
# print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
# print('Inference:{:s}'.format(jud))
# #样本外检验
# #单因子模型：OLS线性拟合
# factor_out='DP'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','DP','DPL1']].copy(deep=True)#年份,收益率,无风险超额,DP,DPLOG
# n_in=np.sum(datafit['yyyymm']<=195612)#样本内
# n_out=np.sum(datafit['yyyymm']>195612)#样本外
# rout=np.zeros(n_out)#预测的坑
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)#波动率
# for i in range(n_out):#预测nout次
#     model=smf.ols('ExRet~DPL1',data=datafit[['ExRet','DPL1']].iloc[:(n_in+i),:])#往前滚
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['DPL1']
#     f=datafit['DP'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='EP'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','EP','EPL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~EPL1',data=datafit[['ExRet','EPL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['EPL1']
#     f=datafit['EP'].iloc[n_in+i-1]#前一天的因子
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)#最近12个月的平方和
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='VOL'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','VOL','VOLL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~VOLL1',data=datafit[['ExRet','VOLL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['VOLL1']
#     f=datafit['VOL'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='BILL'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','BILL','BILLL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~BILLL1',data=datafit[['ExRet','BILLL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['BILLL1']
#     f=datafit['BILL'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='BOND'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','BOND','BONDL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~BONDL1',data=datafit[['ExRet','BONDL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['BONDL1']
#     f=datafit['BOND'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='TERM'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','TERM','TERML1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~TERML1',data=datafit[['ExRet','TERML1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['TERML1']
#     f=datafit['TERM'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='CREDIT'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','CREDIT','CREDITL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~CREDITL1',data=datafit[['ExRet','CREDITL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['CREDITL1']
#     f=datafit['CREDIT'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='MA112'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','MA112','MA112L1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~MA112L1',data=datafit[['ExRet','MA112L1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['MA112L1']
#     f=datafit['MA112'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='MA312'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','MA312','MA312L1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~MA312L1',data=datafit[['ExRet','MA312L1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['MA312L1']
#     f=datafit['MA312'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='MOM6'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','MOM6','MOM6L1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~MOM6L1',data=datafit[['ExRet','MOM6L1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['MOM6L1']
#     f=datafit['MOM6'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='PPIG'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','PPIG','PPIGL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~PPIGL1',data=datafit[['ExRet','PPIGL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['PPIGL1']
#     f=datafit['PPIG'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# factor_out='IPG'
# datafit=data[['yyyymm','Ret','Rfree','ExRet','IPG','IPGL1']].copy(deep=True)
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
# for i in range(n_out):
#     model=smf.ols('ExRet~IPGL1',data=datafit[['ExRet','IPGL1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     b=results.params['Intercept']
#     k=results.params['IPGL1']
#     f=datafit['IPG'].iloc[n_in+i-1]
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in + i]
#     rout[i]=k*f+b
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
# print()
# print('Out-of-sample tests for one factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
# #样本外检验
# #多因子模型：OLS线性拟合
# factor_out='DP,EP,VOL,BILL,BOND,TERM,CREDIT,PPIG,IPG,MA112,MA312,MOM6'
# datafit=data.copy(deep=True)
#
# n_in=np.sum(datafit['yyyymm']<=195612)
# n_out=np.sum(datafit['yyyymm']>195612)
# rout=np.zeros(n_out)
# rmean=np.zeros(n_out)
# rreal=np.zeros(n_out)
# rfree=np.zeros(n_out)
# volt2=np.zeros(n_out)
#
# for i in range(n_out):
#     model=smf.ols('ExRet~DPL1+EPL1+VOLL1+BILLL1+BONDL1+TERML1+CREDITL1+'
#                   'PPIGL1+IPGL1+MA112L1+MA312L1+MOM6L1',
#                   data=datafit[['ExRet','DPL1','EPL1','VOLL1','BILLL1','BONDL1','TERML1',
#                                 'CREDITL1','PPIGL1','IPGL1','MA112L1','MA312L1','MOM6L1']].iloc[:(n_in+i),:])
#     results=model.fit()
#     k=results.params.values
#     f=datafit[['DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG',
#                'IPG','MA112','MA312','MOM6']].iloc[n_in+i-1,:].values
#     f=np.concatenate((np.array([1]),f))
#     rreal[i]=datafit['ExRet'].iloc[n_in+i]
#     rfree[i]=datafit['Rfree'].iloc[n_in+i]
#     rout[i]=np.sum(k*f)
#     rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
#
# print()
# print('Out-of-sample tests for multi-factor model with OLS:')
# print('Predictor:{:s}'.format(factor_out))
# R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
# Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
# del datafit
#
#
#
# #样本外检验
# #多因子模型：LASSO回归，Ridge回归，ElasticNet回归
# factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
# factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])
#
# datafit = data.copy(deep=True)
#
# n_in = np.sum(datafit['yyyymm'] <= 195612)
# n_out = np.sum(datafit['yyyymm'] > 195612)
# rout = np.zeros(n_out)
# rmean = np.zeros(n_out)
# rreal = np.zeros(n_out)
# rfree = np.zeros(n_out)
# volt2 = np.zeros(n_out)
# #Ridge
# reg = sklm.RidgeCV(cv=10, fit_intercept=True, normalize=True)
# for i in range(n_out):
#     X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
#                  'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
#     y = datafit['ExRet'].iloc[:(n_in+i)].values
#     reg.fit(X, y)
#     # print(factor_list[np.abs(reg.coef_) != 0])
#     k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
#     f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
#                  'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
#     f = np.concatenate((np.array([1]), f))
#     rreal[i] = datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in+i]
#     rout[i] = np.sum(k*f)
#     rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
#
# print()
# print('Out-of-sample tests for multi-factor model with ML method:')
# print('Predictor: {:s}'.format(factor_out))
# R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
# Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
# del datafit
#
# #Lasso
# factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
# factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])
#
# datafit = data.copy(deep=True)
#
# n_in = np.sum(datafit['yyyymm'] <= 195612)
# n_out = np.sum(datafit['yyyymm'] > 195612)
# rout = np.zeros(n_out)
# rmean = np.zeros(n_out)
# rreal = np.zeros(n_out)
# rfree = np.zeros(n_out)
# volt2 = np.zeros(n_out)
# reg = sklm.LassoCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
# # reg_lasso = linear_model.LassoLarsCV(cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10000000)
# # reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
# for i in range(n_out):
#     X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
#                  'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
#     y = datafit['ExRet'].iloc[:(n_in+i)].values
#     reg.fit(X, y)
#     # print(factor_list[np.abs(reg.coef_) != 0])
#     k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
#     f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
#                  'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
#     f = np.concatenate((np.array([1]), f))
#     rreal[i] = datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in+i]
#     rout[i] = np.sum(k*f)
#     rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
#
# print()
# print('Out-of-sample tests for multi-factor model with ML method:')
# print('Predictor: {:s}'.format(factor_out))
# R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
# Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
# del datafit
#
# #ElasticNet
# factor_out = 'DP, EP, VOL, BILL, BOND, TERM, CREDIT, PPIG, IPG, MA112, MA312, MOM6'
# factor_list = np.array(['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6'])
#
# datafit = data.copy(deep=True)
#
# n_in = np.sum(datafit['yyyymm'] <= 195612)
# n_out = np.sum(datafit['yyyymm'] > 195612)
# rout = np.zeros(n_out)
# rmean = np.zeros(n_out)
# rreal = np.zeros(n_out)
# rfree = np.zeros(n_out)
# volt2 = np.zeros(n_out)
# reg = sklm.ElasticNetCV(random_state=0, cv=10, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, n_jobs=-1, max_iter=10**9, tol=10-6)
# for i in range(n_out):
#     X = datafit[['DPL1', 'EPL1', 'VOLL1', 'BILLL1', 'BONDL1', 'TERML1',
#                  'CREDITL1', 'PPIGL1', 'IPGL1', 'MA112L1', 'MA312L1', 'MOM6L1']].iloc[:(n_in+i), :].values
#     y = datafit['ExRet'].iloc[:(n_in+i)].values
#     reg.fit(X, y)
#     # print(factor_list[np.abs(reg.coef_) != 0])
#     k = np.concatenate((np.array([reg.intercept_]), reg.coef_))
#     f = datafit[['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG',
#                  'IPG', 'MA112', 'MA312', 'MOM6']].iloc[n_in+i-1, :].values
#     f = np.concatenate((np.array([1]), f))
#     rreal[i] = datafit['ExRet'].iloc[n_in+i]
#     rfree[i] = datafit['Rfree'].iloc[n_in+i]
#     rout[i] = np.sum(k*f)
#     rmean[i] = np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
#     volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)
#
# print()
# print('Out-of-sample tests for multi-factor model with ML method:')
# print('Predictor: {:s}'.format(factor_out))
# R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
# Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
# del datafit
#第三题
data2=pd.read_csv('实验4（3）数据.csv',encoding='GB2312')#676一个
#beta数据
beta=pd.read_csv('beta数据.csv',encoding='GB2312')
beta=pd.DataFrame(beta)
beta.columns=['Stkcd','日期_Date','Beta']
beta['日期_Date']=pd.to_datetime(beta['日期_Date'])

data2=pd.DataFrame(data2)
data2['日期_Date']=pd.to_datetime(data2['日期_Date'])
data2['yearmonth'] = data2['日期_Date'].dt.strftime('%Y%m')#记得先转换成datime
print(data2)
#代码 日期 收盘价
daily1=pd.read_csv('日度数据1.csv',encoding='GB2312')#2001-2010
daily1=pd.DataFrame(daily1)
daily2=pd.read_csv('日度数据2.csv',encoding='GB2312')#2010-2015
daily2=pd.DataFrame(daily2)
daily3=pd.read_csv('日度数据3.csv',encoding='GB2312')#2015-2020
daily3=pd.DataFrame(daily3)
daily4=pd.read_csv('日度数据4.csv',encoding='GB2312')#2020-2021
daily4=pd.DataFrame(daily4)
dailydata=pd.concat([daily1,daily2,daily3,daily4],axis=0,join='outer')#拼在下面
# axis=0 表示按行连接，axis=1 表示按列连接；join 参数指定合并方式
dailydata.columns=['Stkcd','date','Clpr','4']
# # 月股价高点（当月股价最高值与前三个 月股价的最大值的比值，用日度数据计算）
# TypeError: first argument must be an iterable of pandas objects, you passed an object of type "DataFrame"
# 出错原因就是，在使用pandas.concat(a,b)进行合并的时候，需要是list的形式。因此改成pandas.concat([a,b]),就可以成功合并
#删除列
dailydata.drop(columns=['4'],axis=0,inplace=True)
dailydata['date']=pd.to_datetime(dailydata['date'])
dailydata['yearmonth'] = dailydata['date'].dt.strftime('%Y%m')#记得先转换成datime
dailydata.index=dailydata['date']#设置所以索引
s=np.unique(dailydata['yearmonth'].values)
high=np.zeros(len(s))#月数量
# # 重要复习月股价高点（当月股价最高值与前三个 月股价的最大值的比值，用日度数据计算）
#整行数据能使用
for i in range(len(s)):
    a=dailydata[dailydata['yearmonth']==s[i]]#属于每个月的数据
    high[i]=max(a.iloc[:,2].values)#收盘价数据
print(dailydata)
high_data=pd.DataFrame(high,columns=['high'])
print(high_data)#每个月的月股价高点

# print(len(s))#252
dailydata.dropna(inplace=True)
dailydata.columns=['Stkcd','date','Clpr','yearmonth']
k=[]
# .iloc根据行号索引3grouppby函数
df_grouped = dailydata.groupby(['yearmonth'])['Clpr'].std().reset_index()#月波动率
print(df_grouped)#出来只有两列['yearmonth']['Clpr']
df_grouped.columns=['yearmonth','month_vol']
dailydata = pd.merge(dailydata, df_grouped, on='yearmonth', how='inner')#将波动率拼回去日度数据
print(dailydata)
dailydata.set_index('date', inplace=True)#日度数据转为月度数据
dailydata2=dailydata.resample('M').last()#日度数据转为月度数据
dailydata2.reset_index()#日度数据转为月度数据
print(dailydata2)#日度数据转为月度数据
#新增date列
dailydata2['date']=dailydata2.index
dailydata2['yearmonth'] = dailydata2['date'].dt.strftime('%Y%m')#记得先转换成datime
print(dailydata2)
# dailydata2.to_excel('日度.xlsx')
#拼回去上面的月度数据
matrix=pd.merge(left=data2,right=dailydata2[['yearmonth','month_vol']],
                    on='yearmonth',how='inner',sort=True)

print(matrix)#拼起来
matrix['high']=high_data
high_point=matrix['high'].values
for i in range(3,252):
    high_point[i]=(high[i]/max(high[i-1],high[i-2],high[i-3]))

high_point[0]= 26.68/27.07
high_point[1]=27.07/27.47
high_point[2]=27.47/26.62
matrix['high']=high_point
print(matrix)
from scipy.stats import skew





# # # 登陆系统
# # # lg = bs.login()
# # # # 显示登陆返回信息
# # # print(lg.error_code)
# # # print(lg.error_msg)
# # # rs = bs.query_history_k_data("sz.000676",
# # #     "date,code,close",
# # #     start_date='2001-01-01', end_date='2021-12-31',
# # #     frequency='5', adjustflag="3")
# # # print(rs.error_code)
# # # print(rs.error_msg)
# # # # 获取具体的信息
# # # result_list = []
# # # while (rs.error_code == '0') & rs.next():
# # #     # 分页查询，将每页信息合并在一起
# # #     result_list.append(rs.get_row_data())
# # # result = pd.DataFrame(result_list, columns=rs.fields)
# # # bs.logout()
# # # result.to_excel('日内数据.xlsx')
result=pd.read_excel('日内数据.xlsx')
result['date']=pd.to_datetime(result['date'])
result['yearmonthdate']=result['date'].dt.strftime('%Y%m%d').astype(int)
result['return']=np.log(result['close']/result['close'].shift(1))
result = result.drop(result.columns[0], axis=1)#删除第一列
result.dropna(inplace=True)
result.index=result['date']
result['return_sq'] = result['return'] ** 2
# 计算每个月的实现偏度
skews = result.groupby(pd.Grouper(freq='M'))['close'].apply(skew)
# 输出结果
print(skews)
skews=pd.DataFrame(skews)
skews['日期']=skews.index
skews['日期']=pd.to_datetime(skews['日期'])
skews.columns=['月偏度','日期']
skews['yearmonth'] = skews['日期'].dt.strftime('%Y%m')#记得先转换成datime
print(skews)
# 使用resample函数将日度数据转换为月度数据
# result_monthly = result.resample('M').mean()
# print(result_monthly)
matrix2=pd.merge(left=matrix,right=skews[['yearmonth','月偏度']], on='yearmonth',how='inner',sort=True)
matrix2=pd.merge(left=matrix2,right=beta[['日期_Date','Beta']], on='日期_Date',how='inner',sort=True)
print(matrix2)
# matrix2.to_excel('data数据.xlsx')
print(matrix2.columns)

matrix2.columns=[['Stkcd','date','close','VOL','Trdsum','Turnover','Ret','Rfree','PE',
                'EPS','ROE','IncomePS','yyyymm','波动率','high','rskew','Beta']]
print(matrix2)
matrix2.dropna(inplace=True)
# 月流动性（|月收益率 / lg(月成交额)|
matrix2['liqulity']=abs((matrix2['Ret'].values/np.log(matrix2['Trdsum']).values))#注意一下一开始有报错
print(matrix2)
matrix2['ExRet']=matrix2['Ret'].values-matrix2['Rfree'].values
data=pd.concat([matrix2[['VOL','Trdsum','Turnover','Ret','Rfree','PE',
                'EPS','ROE','IncomePS','yyyymm','波动率','high','rskew','Beta','liqulity','ExRet']],
                matrix2[['VOL','Trdsum','Turnover','Ret','Rfree','PE',
                'EPS','ROE','IncomePS','波动率','high','rskew','Beta','liqulity','ExRet']].shift(periods=1)],axis=1)#按列也就是右边
data.columns=['VOL','Trdsum','Turnover','Ret','Rfree','PE',
                'EPS','ROE','IncomePS','yyyymm','波动率','high','rskew','Beta','liqulity','ExRet',
                    'VOL1','Trdsum1','Turnover1','Ret1','Rfree1','PE1',
                'EPS1','ROE1','IncomePS1','波动率1','high1','rskew1','Beta1','liqulity1','ExRet1']
data['date']=pd.to_datetime(data['yyyymm'],format='%Y%m')
data.reset_index(drop=True,inplace=True)
data.dropna(inplace=True)
print(data)
data.to_excel('datacharles.xlsx')
def myfun_stat_gains(rout,rmean,rreal):
    R2os=1-np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)
    d=(rreal-rmean)**2-((rreal-rout)**2-(rmean-rout)**2)#是y
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
#统计显著性函数
def myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5):
    omg_out=rout/volt2/gmm#rout 风险偏好 rout 波动率 gmm风险偏好==》t时刻资产配置的比率
    rp_out=rfree+omg_out*rreal#组合投资收益率
    Uout=np.mean(rp_out)-0.5*gmm*np.var(rp_out)#计算投资效用
    #以下考量均值模型
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
factor='VOL'
# smf跟sm差别在于smf可以直接指定谁跟谁的关系
model=smf.ols('ExRet~VOL1',data=data[['ExRet','VOL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['VOL1']
rg_DP_pvalue=results.pvalues['VOL1']
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
print('Regressing Results:b={:f},k={:f}'.format(rg_con,rg_DP))#b是截距 k是斜率
print('Regressing Results:p={:f},p={:f}'.format(rg_con_pvalue,rg_DP_pvalue))
print('Inference:{:s}'.format(jud))

factor='Trdsum'
model=smf.ols('ExRet~Trdsum1',data=data[['ExRet','Trdsum1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['Trdsum1']
rg_DP_pvalue=results.pvalues['Trdsum1']
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
model=smf.ols('ExRet~Turnover1',data=data[['ExRet','Turnover1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['Turnover1']
rg_DP_pvalue=results.pvalues['Turnover1']
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


factor='Rfree'
model=smf.ols('ExRet~Rfree1',data=data[['ExRet','Rfree1']])
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['Rfree1']
rg_DP_pvalue=results.pvalues['Rfree1']
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

factor='PE'
model=smf.ols('ExRet~PE1',data=data[['ExRet','PE1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['PE1']
rg_DP_pvalue=results.pvalues['PE1']
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
model=smf.ols('ExRet~EPS1',data=data[['ExRet','EPS1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['EPS1']
rg_DP_pvalue=results.pvalues['EPS1']
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
model=smf.ols('ExRet~ROE1',data=data[['ExRet','ROE1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['ROE1']
rg_DP_pvalue=results.pvalues['ROE1']
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
model=smf.ols('ExRet~IncomePS1',data=data[['ExRet','IncomePS1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['IncomePS1']
rg_DP_pvalue=results.pvalues['IncomePS1']
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

factor='波动率'
model=smf.ols('ExRet~波动率1',data=data[['ExRet','波动率1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['波动率1']
rg_DP_pvalue=results.pvalues['波动率1']
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
model=smf.ols('ExRet~high1',data=data[['ExRet','high1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['high1']
rg_DP_pvalue=results.pvalues['high1']
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
model=smf.ols('ExRet~rskew1',data=data[['ExRet','rskew1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['rskew1']
rg_DP_pvalue=results.pvalues['rskew1']
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
model=smf.ols('ExRet~Beta1',data=data[['ExRet','Beta1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['Beta1']
rg_DP_pvalue=results.pvalues['Beta1']
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
model=smf.ols('ExRet~liqulity1',data=data[['ExRet','liqulity1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['liqulity1']
rg_DP_pvalue=results.pvalues['liqulity1']
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
data['yyyymm']=data['yyyymm'].astype('int64')
# #样本外检验
# #单因子模型：OLS线性拟合
factor_out='VOL'
datafit=data[['yyyymm','Ret','Rfree','ExRet','VOL','VOL1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~VOL1',data=datafit[['ExRet','VOL1']].iloc[:(n_in+i),:])#往前滚先用n_in数据预测n_in+i的数据
    results=model.fit()
    b=results.params['Intercept']#截距
    k=results.params['VOL1']#斜率
    f=datafit['VOL'].iloc[n_in+i-1]#因子值
    rreal[i]=datafit['ExRet'].iloc[n_in+i]#真实收益率
    rfree[i] = datafit['Rfree'].iloc[n_in + i]#无风险利率
    rout[i]=k*f+b
    rmean[i]=np.mean(datafit['ExRet'].iloc[:(n_in+i)].values)
    volt2[i]=np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)#波动率最近12个月的收益率平方和
print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor:{:s}'.format(factor_out))
R2os,MFSEadj,pvalue_MFSEadj=myfun_stat_gains(rout,rmean,rreal)
Uout,Umean,DeltaU=myfun_econ_gains(rout,rmean,rreal,rfree,volt2,gmm=5)
del datafit
# 'Turnover','Ret','Rfree','PE',
# #                 'EPS','ROE','IncomePS','yyyymm','波动率','high','rskew','Beta','liqulity','ExRet'

factor_out='Trdsum'
datafit=data[['yyyymm','Ret','Rfree','ExRet','Trdsum','Trdsum1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201011)
n_out=np.sum(datafit['yyyymm']>201011)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~Trdsum1',data=datafit[['ExRet','Trdsum1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['Trdsum1']
    f=datafit['Trdsum'].iloc[n_in+i-1]
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
datafit=data[['yyyymm','Ret','Rfree','ExRet','Turnover','Turnover1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~Turnover1',data=datafit[['ExRet','Turnover1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['Turnover1']
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
# #                 'EPS1','ROE1','IncomePS1','波动率1','high1','rskew1','Beta1','liqulity1','ExRet1'
factor_out='PE'
datafit=data[['yyyymm','Ret','Rfree','ExRet','PE','PE1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~PE1',data=datafit[['ExRet','PE1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['PE1']
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
# 'EPS1','ROE1','IncomePS1','波动率1','high1','rskew1','Beta1','liqulity1','ExRet1'
factor_out='EPS'
datafit=data[['yyyymm','Ret','Rfree','ExRet','EPS','EPS1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~EPS1',data=datafit[['ExRet','EPS1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['EPS1']
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
# ,'IncomePS1','波动率1','high1','rskew1','Beta1','liqulity1','ExRet1'
factor_out='ROE'
datafit=data[['yyyymm','Ret','Rfree','ExRet','ROE','ROE1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~ROE1',data=datafit[['ExRet','ROE1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['ROE1']
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
datafit=data[['yyyymm','Ret','Rfree','ExRet','IncomePS','IncomePS1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~IncomePS1',data=datafit[['ExRet','IncomePS1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['IncomePS1']
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

factor_out='波动率'
datafit=data[['yyyymm','Ret','Rfree','ExRet','波动率','波动率1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=201012)
n_out=np.sum(datafit['yyyymm']>201012)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~波动率1',data=datafit[['ExRet','波动率1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['波动率1']
    f=datafit['波动率'].iloc[n_in+i-1]
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
datafit=data[['yyyymm','Ret','Rfree','ExRet','high','high1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~high1',data=datafit[['ExRet','high1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['high1']
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
# ,'liqulity1','ExRet1'
factor_out='rskew'
datafit=data[['yyyymm','Ret','Rfree','ExRet','rskew','rskew1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~rskew1',data=datafit[['ExRet','rskew1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['rskew1']
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
factor_out='Beta'
datafit=data[['yyyymm','Ret','Rfree','ExRet','Beta','Beta1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~Beta1',data=datafit[['ExRet','Beta1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['Beta1']
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
datafit=data[['yyyymm','Ret','Rfree','ExRet','liqulity','liqulity1']].copy(deep=True)
n_in=np.sum(datafit['yyyymm']<=200512)
n_out=np.sum(datafit['yyyymm']>200512)
rout=np.zeros(n_out)
rmean=np.zeros(n_out)
rreal=np.zeros(n_out)
rfree=np.zeros(n_out)
volt2=np.zeros(n_out)
for i in range(n_out):
    model=smf.ols('ExRet~liqulity1',data=datafit[['ExRet','liqulity1']].iloc[:(n_in+i),:])#往前滚
    results=model.fit()
    b=results.params['Intercept']
    k=results.params['liqulity1']
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
    model=smf.ols('ExRet~PE1+Turnover1+VOL1+EPS1+ROE1+IncomePS1+Beta1+liqulity1+high1+rskew1',
                  data=datafit[['ExRet','PE1','Turnover1','VOL1','EPS1','ROE1','IncomePS1','Beta1','liqulity1','high1','rskew1']].iloc[:(n_in+i),:])
    results=model.fit()
    k=results.params.values
    f=datafit[['PE','Turnover','VOL','EPS','ROE','IncomePS','Beta','liqulity','high','rskew']].iloc[n_in+i-1,:].values#因子数
    f=np.concatenate((np.array([1]),f))#将常数项放进来
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
#可能存在多重共线性，使预测能力下降

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
reg = sklm.RidgeCV(cv=10, fit_intercept=True, normalize=True)#sklear的包
for i in range(n_out):
    X = datafit[['PE1','Turnover1','VOL1','EPS1','ROE1','IncomePS1','Beta1','liqulity1','high1','rskew1']].iloc[:(n_in+i), :].values
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
    X = datafit[['PE1','Turnover1','VOL1','EPS1','ROE1','IncomePS1','Beta1','liqulity1','high1','rskew1']].iloc[:(n_in+i), :].values
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
    X = datafit[['PE1','Turnover1','VOL1','EPS1','ROE1','IncomePS1','Beta1','liqulity1','high1','rskew1']].iloc[:(n_in+i), :].values
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

