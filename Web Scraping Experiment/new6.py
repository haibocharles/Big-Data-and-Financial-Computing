import numpy as np
import pandas as pd
import time
import requests
import re
import io
import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
from lxml import etree  # 解析文档
import requests  # 获取网页
import pandas as pd  # 保存文件
import time
import random
from faker import Factory
import re
import numpy as np
head = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",'Cookie':'bid=rH_pO1e8240;__utmc=30149280;__utmc=223695111; ll="108296";dbcl2="183603207:5XY9kAopCdc"; ck=NRfL;_pk_ref.100001.4cf6=["","",1646313803,"https://accounts.douban.com/"];_pk_ses.100001.4cf6=*;__utma=30149280.1438788507.1646053256.1646217497.1646313803.4;__utmb=30149280.0.10.1646313803;__utmz=30149280.1646313803.4.2.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/;__utma=223695111.1480997948.1646053256.1646217497.1646313803.3;__utmb=223695111.0.10.1646313803;__utmz=223695111.1646313803.3.2.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/;push_noty_num=0;push_doumail_num=0;_pk_id.100001.4cf6=bc97690c3902772b.1646053254.3.1646313815.1646217503.'
}
def get_html(i):
    url = 'https://guba.eastmoney.com/list,603569_' + str(i) + '.html'
    resp = requests.get(url, headers=head, timeout=100)
    html = resp.text
    print(html)
    data = get_data(html)
    print(data)
#     write_data(data)
    print(f"已爬取第{i}页")
