## Baidu_translate_study
You can create your account at [link](http://api.fanyi.baidu.com/) to get vip translate service, Baidu will provide a **appid** and **secretKey** for you to build your own translate service in your program, this template interface program can be used in your customized application.

### Main Process
- build connection
- get response
- parse data

### Important functions
- http.client.HTTPConnection('api.fanyi.baidu.com')
- httpClient.request('GET', myurl)
- response = httpClient.getresponse()
- result_all = response.read().decode("utf-8")
- result = json.loads(result_all)

### Special code
```python
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
```