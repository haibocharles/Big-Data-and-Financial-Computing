import numpy as np
import pandas as pd
import time
import requests
import re
import io
import sys
import statsmodels as sm
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.50'}
url=r'https://guba.eastmoney.com/o/list,603569.html'
res=requests.get(url,headers=headers,timeout=30)
res.encoding='utf-8'
html=res.text
print(html)
# <div class="pager" style="border-top: 1px solid #D7E5FF;">
#                     <span class="pagernums" data-pager="list,603569_|30176|80|1"><span class="pagerbox"><span><a class="first_page" data-page="1" href="list,603569_1.html" target="_self">首页</a><a data-page="1" class="on" href="list,603569_1.html" target="_self">1</a><a data-page="2" href="list,603569_2.html" target="_self">2</a><a data-page="3" href="list,603569_3.html" target="_self">3</a><a data-page="4" href="list,603569_4.html" target="_self">4</a><a data-page="5" href="list,603569_5.html" target="_self">5</a><a data-page="6" href="list,603569_6.html" target="_self">6</a><a data-page="7" href="list,603569_7.html" target="_self">7</a><a data-page="8" href="list,603569_8.html" target="_self">8</a><a data-page="9" href="list,603569_9.html" target="_self">9</a><a data-page="2" href="list,603569_2.html" target="_self">下一页</a><a class="last_page" data-page="378" href="list,603569_378.html" target="_self">尾页</a> 共<span class="sumpage">378</span>页</span><span class="jump_page">跳转至 <input class="jump_input"> 页</span><span class="jump_launch">确定</span></span></span>
#
#                 </div>
# 第三题
# def myfun_crawl_sina_news():
#     url = 'http://guba.eastmoney.com/list,603569_'
#
#     pcheck = r'<div class="dheader">.+?<div class="gbbox1" id="sendnewt">'
#     p = r'<div class="articleh normal_post">.*?<span class="l1 a1">(.+?)</span>.*?<span class="l2 a2">(.+?)</span>.*?<span class="l3 a3"><a\shref="(.+?)" title="(.+?)">.+?</a></span>.*?<span class="l4 a4">.+?target="_blank"><font>(.+?)</font></a>.+?</span>.*?<span class="l5 a5">(.+?)</span>.*?</div>'
#     objp = re.compile(p, re.DOTALL)
#
#     fw = open('data_guba_cjwl.txt', 'w', encoding='utf-8')
#     for i in range(1, 350):  # 爬取1-5页的数据
#         print(i)
#         headers = {'User-Agent': fc.user_agent()}
#         while True:
#             try:
#                 res = requests.get(url + str(i) + '.html', headers=headers, timeout=10)  # 读数据
#                 res.encoding = 'utf-8'#改编码，防止乱码
#                 html = res.text
#                 mcheck = re.search(pcheck, html, re.DOTALL)
#                 if len(mcheck.group()) > 0:#mcheck包含数据则大于0
#                     html = mcheck.group()
#                     break
#             except:
#                 print('failing to crawl the data because of timeout')
#             time.sleep(np.random.randint(10, 15))
#         match = objp.findall(html)
#         for line in match:
#             print(line)
#             fw.write('{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n'.format(line[0], line[1], line[2],line[3], line[4], line[5]))#\n表示换行
#     fw.close()
# myfun_crawl_sina_news()



# 第三题
# '<span class="l1 a1">(.*?)</span>.*?<span class="l2 a2">(.*?)</span>.*?<span class="l3 a3">.*?<a.*?>(.*?)</a></span>.*?<span class="l4 a4">.*?<a.*?><font>(.*?)</font></a>.*?/span>.*?<span class="l5 a5">(.*?)</span>'
# fw = open('Data_SinaNews.txt', 'w',encoding='utf-8')
# for i in range(1,377,1):
#     currenturl= r'https://guba.eastmoney.com/o/list,603569_{}.html'.format(i)
# # pri<div class="articleh normal_post odd">
#     response=requests.get(currenturl,headers=headers)
#     content=response.content.decode(encoding='utf-8', errors='ignore')
#         # read='<span class="l1 a1">(.*?)</span>'
#         # review=re.findall('<span class="l2 a2">(.*?)</span>',content,re.DOTALL)
#     titile='<span class="l1 a1">(.*?)</span>.*?<span class="l2 a2">(.*?)</span>.*?<span class="l3 a3">.*?<a.*?>(.*?)</a></span>.*?<span class="l4 a4">.*?<a.*?><font>(.*?)</font></a>.*?/span>.*?<span class="l5 a5">(.*?)</span>'
#     # author=re.findall('<span class="l4 a4">(.*?)/span>',content,re.DOTALL)
#     # time=re.findall('<span class="l5 a5">(.*?)</span>',content,re.DOTALL)
#     objp = re.compile(titile, re.S)
#     match = objp.findall(content)
#     for line in match:
#         fw.write('\n{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t\n'.format(line[0],line[1],line[2],line[3],line[4]))
#         fw.write('\n')
#     print(match)
# fw.close()

# 第二题
#存储数据

# p1 = r'<tr.+?ter">.+?(\d{4}-\d{2}-\d{2}).+?</div.+?ter">(.+?)</div.+?ter">(.+?)</div.+?ter">(.+?)</div>.+?ter">(.+?)</div.+?ter">(.+?)</div.+?ter">(.+?)</div>'
#
# fw = open('data.csv', 'w')
# for nian in range(1999, 2019):
#     for jidu in range(1, 5):
#         with open('DataHTML_600618_Year_' + str(nian) + '_Jidu_' + str(jidu) + '.txt', 'r', encoding='utf-8') as file:
#             html = file.read()
#             match = re.findall(p1, html, re.S)
#             if match:
#                 for line in match:
#                     fw.write('{:s}, {:s}, {:s}, {:s}, {:s}, {:s}, {:s}\n'.format(line[0], line[1], line[2], line[3],
#                                                                                  line[4], line[5], line[6]))
# fw.close()


#第四题
# res = requests.get('http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh603569&scale=240&ma=no&datalen=10000')
# data_json = res.json()
# fw = open('data_sina_api.txt', 'w')
# fw.write('day, open, high, low, close, volume\n')
# for i in range(len(data_json)):
#     dj = data_json[i]
#     fw.write('{:s},{:s},{:s},{:s},{:s},{:s}\n'.format(dj['day'], dj['open'], dj['high'],
#                                                     dj['low'], dj['close'], dj['volume']))
# fw.close()


# import pandas as pd







# #第五题
# data = pd.read_csv('Data_SinaNews.txt', header=None, names=['Reading_volume','comment','postlink','title','author','time'])
# print(data)
# data=pd.DataFrame(data)
# data.columns=['Reading_volume','comment','postlink','title','author','time']
# data['title len']=data['title']
# data['author len']=data['author']
# posttime=data['time']
# data['title len'] = data['title'].apply(lambda x: len(str(x)))
# data['author len'] = data['author'].apply(lambda x: len(str(x)))
# data['time'] = data['time'].apply(lambda x: str(x)[1:5])
# database=pd.concat((data.iloc[:,5],data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,6],data.iloc[:,7]),axis=1)
# print(database)
# database.to_excel('database.xlsx')














#
shouyi=pd.read_csv('600618日读数据.csv',encoding='GB2312')
shouyi.dropna(inplace=True)
shouyi['日期_Date']=pd.to_datetime(shouyi['日期_Date'])
# 股票代码_Stkcd	日期_Date	日收益率_Dret	日无风险收益率_DRfRet
print(shouyi)

shouyi.columns=['stkcd','日期','ret','rfret']
shouyi['exret']=shouyi['ret']-shouyi['rfret']
shouyi['日期']=shouyi['日期'].dt.strftime('%m-%d')
print(shouyi)
# # # # #
# # # # #
# # # # # # 读取数据文件
df = pd.read_csv("data_guba_cjwl.txt", sep="\t", names=["阅读量", "评论数",'link', "帖子标题",'作者','时间'], error_bad_lines=False)
# df.index=df['时间']
print(df)
#data = data.drop(['发帖时间'],axis=1)
df.drop(columns=["评论数",'link', "帖子标题",'作者'],axis=1,inplace=True)
print(df)
df= df.groupby(['时间'],as_index=False)['阅读量'].sum()
print(df)
df.columns=['日期','阅读量']
df['日期']=pd.to_datetime(df['日期'],format='%m-%d %H:%M')
df['日期']=df['日期'].dt.strftime('%m-%d')
print(df)
#取平均不然数据太多了
df = df[~df['阅读量'].str.contains('万')]
df['阅读量'] = df['阅读量'].astype(float)
df['阅读量'] = df['阅读量'].fillna(df['阅读量'].mean())
df = df.groupby('日期', as_index=False).agg({'阅读量': 'mean'})
print(df)
data1 = pd.merge(left = shouyi[['日期','ret','exret','rfret']],
               right = df[['日期','阅读量']],
               on = '日期',
               how = 'inner')
print(data1)
data1 = pd.concat([data1[['日期','ret','exret','rfret','阅读量']],
                 data1[['阅读量']].shift(periods=1)],axis=1)
print(data1)
data1.columns = ['date','ret','exret','rfret','view','viewL1']
# data1=data1.sort_values(by='date',inplace=True)
# data1.dropna(inplace=True)
# print(data1)
# data1.dropna(inplace=True)


# data1 = data1.groupby('month', as_index=False).agg({'ret': 'mean', 'exret': 'mean', 'rm': 'mean', 'view': 'mean', 'viewL1': 'mean'})
# # 将聚合后的数据转换为DataFrame对象
# monthly_data = pd.DataFrame(monthly_data)
# monthly_data.replace([np.inf, -np.inf], 4.239982e+209, inplace=True)
# # # 输出月度数据
# print(monthly_data)

import statsmodels.formula.api as smf
import statsmodels.api as sm
# 样本外检验函数
# 单因子模型（双变量预测模型）
def myfun_stat_gains(rout, rmean, rreal):  # 预测收益率，均值收益率，实际收益率
    R2os = 1 - np.sum((rreal - rout) ** 2) / np.sum((rreal - rmean) ** 2)
    d = (rreal - rmean) ** 2 - ((rreal - rout) ** 2 - (rmean - rout) ** 2)
    x = sm.add_constant(np.arange(len(d)) + 1)
    model = sm.OLS(d, x)
    fitres = model.fit()
    MFSEadj = fitres.tvalues[0]
    pvalue_MFSEadj = fitres.pvalues[0]

    if (R2os > 0) & (pvalue_MFSEadj <= 0.01):
        jud = '在1%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MFSEadj > 0.01) & (pvalue_MFSEadj <= 0.05):
        jud = '在5%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MFSEadj > 0.05) & (pvalue_MFSEadj <= 0.1):
        jud = '在10%的显著性水平下有样本外预测能力'
    else:
        jud = '无样本外预测能力'
    print('Stat gains: R2os = {:f}, MFSEadj = {:f}, MFSEpvalue = {:f}'.format(R2os, MFSEadj, pvalue_MFSEadj))
    print('Inference: {:s}'.format(jud))

    return R2os, MFSEadj, pvalue_MFSEadj


# Out-of-sample tests 经济显著性检验
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


# 样本内检验：单因子模型: OLS线性拟合
# 将所有因子循环进行单因子检验
factor='view'
model=smf.ols('exret~viewL1',data=data1[['exret','viewL1']])#可以指定回归模型是什么样子
results=model.fit()
rg_con=results.params['Intercept']
rg_con_pvalue=results.pvalues['Intercept']
rg_DP=results.params['viewL1']
rg_DP_pvalue=results.pvalues['viewL1']
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



# 样本外检验
# 单因子模型: OLS线性拟合
factor_out = '阅读量'
datafit = data1[['date', 'ret', 'rfret', 'exret', 'view', 'viewL1']].copy(deep=True)

n_in = 500#注意这里
n_out = len(datafit.iloc[:, 0]) -500#注意这里
rout = np.zeros(n_out)
rmean = np.zeros(n_out)
rreal = np.zeros(n_out)
rfree = np.zeros(n_out)
volt2 = np.zeros(n_out)

for i in range(n_out):
    model = smf.ols('exret ~ viewL1', data=datafit[['exret', 'viewL1']].iloc[:(n_in + i), :])
    results = model.fit()
    b = results.params['Intercept']
    k = results.params['viewL1']
    f = datafit['view'].iloc[n_in + i - 1]
    rreal[i] = datafit['exret'].iloc[n_in + i]
    rfree[i] = datafit['rfret'].iloc[n_in + i]
    rout[i] = k * f + b
    rmean[i] = np.mean(datafit['exret'].iloc[:(n_in + i)].values)
    volt2[i] = np.sum(datafit['ret'].iloc[(n_in + i - 12):(n_in + i)].values ** 2)

print()
print('Out-of-sample tests for one factor model with OLS:')
print('Predictor: {:s}'.format(factor_out))
R2os, MFSEadj, pvalue_MFSEadj = myfun_stat_gains(rout, rmean, rreal)
Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)
del datafit
# 样本外检验
# 多因子模型：OLS线性拟合
#假设输出name和table
string = '<!comment><tr nam="7507"></tr><table>Default</table><br>'
# pattern = r'<(\w+).*?>.*?</\1>'
pattern =r'<.*?<(\w+).*?></tr><(\w+).*?<br>'
output = re.findall(pattern, string)
print('output = ', output)





