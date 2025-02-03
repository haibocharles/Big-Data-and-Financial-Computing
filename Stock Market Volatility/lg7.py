import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({"font.family": "Microsoft YaHei",
"font.size": 20,
"mathtext.fontset": "cm"})
plt.rcParams['font.sans-serif'] = ['SimSun'] # windows
plt.rcParams['axes.unicode_minus'] = False

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
stkcd=np.unique(data['stkcd'].values)
print(stkcd)
data1 = data[data['stkcd'] == stkcd[0]]
data2 = data[data['stkcd'] == stkcd[1]]
p_index_0 = index_0.loc[:, ['close']].values
r_index_0 = np.log(p_index_0[1:]) - np.log(p_index_0[:-1])


p_index_1 = index_1.loc[:, ['close']].values
r_index_1 = np.log(p_index_1[1:]) - np.log(p_index_1[:-1])


p_data1 = data1.loc[:, ['close']].values
r_data1 = np.log(p_data1[1:]) - np.log(p_data1[:-1])


p_data2 = data2.loc[:, ['close']].values
r_data2 = np.log(p_data2[1:]) - np.log(p_data2[:-1])


mask_1 = np.isnan(r_data1)
r_data1 = r_data1[np.logical_not(mask_1)]

mask_2 = np.isnan(r_data2)
r_data2 = r_data2[np.logical_not(mask_2)]

def arch_test(r):
    ret_demean = r - r.mean()
    t3 = het_arch(resid = ret_demean,nlags=3)
    t5 = het_arch(resid = ret_demean,nlags=5)
    return list((t3[1],t5[1]))
print(arch_test(r_index_0))
print(arch_test(r_index_1))
print(arch_test(r_data1))
print(arch_test(r_data2))
#result_arch_test_index_0 = het_arch(r_index_0, nlags=10)
#print(result_arch_test_index_0)


#lagrange_mult , pvalue , f_pvalue ,_, = het_arch(r_index_0)
#print(pvalue)

#lagrange_mult , pvalue , f_pvalue ,_, = het_arch(r_index_1)
#print(pvalue)

#lagrange_mult , pvalue , f_pvalue ,_, = het_arch(r_data1)
#print(pvalue)

#lagrange_mult , pvalue , f_pvalue ,_, = het_arch(r_data2)
#print(pvalue)
def myfun_LG_describe(r):
    print('\n交易天数为：',len(r))
    print('\n均值为：',np.mean(r))
    print('\n标准差为：',np.std(r))
    print('\n偏度为：',stats.skew(r))
    print('\n峰度为：',stats.kurtosis(r))
    print('\n最大值为：',np.max(r))
    print('\n最小值为：', np.min(r))
    r = pd.DataFrame(r)
    print('\n自相关系数为：',r[0].autocorr())
r_index_0_abs = abs(r_index_0)
r_index_1_abs = abs(r_index_1)
r_data1_abs = abs(r_data1)
r_data2_abs = abs(r_data2)
print('上证综指收益率的描述性统计如下：')
print(myfun_LG_describe(r_index_0))

print('上证综指收益率绝对值的描述性统计如下：')
print(myfun_LG_describe(r_index_0_abs))

print('深州成指收益率的描述性统计如下：')
print(myfun_LG_describe(r_index_1))

print('深州成指收益率绝对值的描述性统计如下：')
print(myfun_LG_describe(r_index_1_abs))

print('000650收益率的描述性统计如下：')
print(myfun_LG_describe(r_data1))

print('000650收益率绝对值的描述性统计如下：')
print(myfun_LG_describe(r_data1_abs))

print('600650的描述性统计如下：')
print(myfun_LG_describe(r_data2))

print('600650收益率绝对值的描述性统计如下：')
print(myfun_LG_describe(r_data2_abs))

def myfun_LG(r):
    # mu常数，波动率 ARCH(1, 1)，分布 正态
    am_arch = arch_model(r, mean='constant', vol='arch', p=1, q=1, dist='normal')
    res_arch = am_arch.fit()
    print(res_arch.summary())


    # mu常数，波动率 GARCH(1, 1)，分布 正态
    am_garch = arch_model(r, mean='constant', vol='garch', p=1, q=1, dist='normal')
    res_garch = am_garch.fit()
    print(res_garch.summary())

    # egarch
    am_egarch = arch_model(r * 100, mean='constant', vol='egarch', p=1, q=1, o=1)
    res_egarch = am_egarch.fit()
    print(res_egarch.summary())

    print('ARCH(1) LM test p-value:', res_arch.arch_lm_test().pval)
    print('GARCH(1,1) LM test p-value:', res_garch.arch_lm_test().pval)
    print('EGARCH(1,1) LM test p-value:', res_egarch.arch_lm_test().pval)

    plt.figure(1)
    plt.plot(r)
    sim_arch = arch_model(None, mean='constant', vol='arch', p=1, dist='normal')
    sim_arch_data = sim_arch.simulate(res_arch.params, len(r))
    plt.plot(sim_arch_data.iloc[:, 0].values)
    plt.legend(['原始收益率序列', 'arch 模拟收益率序列'])

    plt.figure(2)
    plt.plot(r)
    sim_garch = arch_model(None, mean='constant', vol='garch', dist='normal', p=1, q=1)

    #sim_paras = pd.Series([0.0006, 0.00008, 0.15, 0.8], index=['mu', 'omega', 'alpha[1]', 'beta[1]'])
    #sim_garch_data = sim_garch.simulate(sim_paras, 1000)
    #plt.plot(sim_garch_data)
    sim_garch_data = sim_garch.simulate(res_garch.params, len(r))
    plt.plot(sim_garch_data.iloc[:, 0].values)
    plt.legend(['原始收益率序列', 'garch 模拟收益率序列'])
    plt.show()

    plt.figure(3)
    plt.plot(r)
    sim_egarch = arch_model(None, mean='constant', vol='egarch', p=1, q=1, o=1, dist='normal')
    sim_egarch_data = sim_egarch.simulate(res_egarch.params, len(r))
    plt.plot(sim_egarch_data.iloc[:, 0].values)
    plt.legend(['原始收益率序列', 'egarch 模拟收益率序列'])
    plt.show()

    print(arch_test(sim_arch_data.iloc[:, 0]))
    print(arch_test(sim_garch_data.iloc[:, 0]))
    print(arch_test(sim_egarch_data.iloc[:, 0]))

myfun_LG(r_index_0)
myfun_LG(r_index_1)
myfun_LG(r_data1)
myfun_LG(r_data2)

