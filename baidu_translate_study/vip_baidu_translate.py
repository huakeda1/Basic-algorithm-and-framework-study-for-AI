#!/usr/bin/env python
# coding: utf-8

# In[1]:


import http.client
import hashlib
import urllib
import random
import json


# In[2]:


class Baidu_Tans(object):
    def __init__(self,fromLang,toLang,appid,secretKey):
        self.fromLang=fromLang
        self.toLang=toLang
        self.appid=appid
        self.secretKey=secretKey
        self.base_url='api.fanyi.baidu.com'
        self.back_url='/api/trans/vip/translate'
    def get_result(self,q):
        httpClient=None
        salt=random.randint(327688,655368)
        sign=self.appid+q+str(salt)+self.secretKey
        sign=hashlib.md5(sign.encode()).hexdigest()
        current_url=self.back_url+'?appid='+self.appid+'&q='+urllib.parse.quote(q)+'&from='+self.fromLang+'&to='+self.toLang+'&salt='+str(salt)+'&sign='+sign
        try:
            httpClient=http.client.HTTPConnection(self.base_url)
            httpClient.request('GET',current_url)
            response=httpClient.getresponse()
            result_all=response.read().decode('utf-8')
            result=json.loads(result_all)['trans_result'][0]['dst']
        except Exception as e:
            print(e)
            result=''
        finally:
            if httpClient:
                httpClient.close()
        return result


# In[3]:


if __name__ == '__main__':
    fromLang='auto'
    toLang='en'
    appid='20201230000659488'
    secretKey='LEd0qCPRWkP8ovRlolih'
    trans=Baidu_Tans(fromLang,toLang,appid,secretKey)
    while(1):
        q=input('please input what you want to translate:')
        if q=='exit':break
        result = trans.get_result(q)
        print(result)


# In[4]:


if __name__ == '__main__':
    fromLang='auto'
    toLang='zh'
    appid='20201230000659488'
    secretKey='LEd0qCPRWkP8ovRlolih'
    trans=Baidu_Tans(fromLang,toLang,appid,secretKey)
    while(1):
        q=input('please input what you want to translate:')
        if q=='exit':break
        result = trans.get_result(q)
        print(result)


# In[5]:


#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json

appid = '20201230000659488'  # 填写你的appid
secretKey = 'LEd0qCPRWkP8ovRlolih'  # 填写你的密钥

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'auto'   #原文语种
toLang = 'zh'   #译文语种 zh or en
while(1):
    salt = random.randint(327688, 655368)
    q=input('please input what you want to translate:')
    if q=='exit':break
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
    salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        print(result['trans_result'][0]['dst'])

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


# ## Baidu_translate_study
# You can create your account at [link](http://api.fanyi.baidu.com/) to get vip translate service, Baidu will provide a **appid** and **secretKey** for you to build your own translate service in your program, this template interface program can be used in your customized application.
# 
# ### Main Process
# - build connection
# - get response
# - parse data
# 
# ### Important functions
# - http.client.HTTPConnection('api.fanyi.baidu.com')
# - httpClient.request('GET', myurl)
# - response = httpClient.getresponse()
# - result_all = response.read().decode("utf-8")
# - result = json.loads(result_all)
# 
# ### Special code
# ```python
# class Baidu_Tans(object):
#     def __init__(self,fromLang,toLang,appid,secretKey):
#         self.fromLang=fromLang
#         self.toLang=toLang
#         self.appid=appid
#         self.secretKey=secretKey
#         self.base_url='api.fanyi.baidu.com'
#         self.back_url='/api/trans/vip/translate'
#     def get_result(self,q):
#         httpClient=None
#         salt=random.randint(327688,655368)
#         sign=self.appid+q+str(salt)+self.secretKey
#         sign=hashlib.md5(sign.encode()).hexdigest()
#         current_url=self.back_url+'?appid='+self.appid+'&q='+urllib.parse.quote(q)+'&from='+self.fromLang+'&to='+self.toLang+'&salt='+str(salt)+'&sign='+sign
#         try:
#             httpClient=http.client.HTTPConnection(self.base_url)
#             httpClient.request('GET',current_url)
#             response=httpClient.getresponse()
#             result_all=response.read().decode('utf-8')
#             result=json.loads(result_all)['trans_result'][0]['dst']
#         except Exception as e:
#             print(e)
#             result=''
#         finally:
#             if httpClient:
#                 httpClient.close()
#         return result
# ```

# In[ ]:




