#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json
import requests


class Trans(object):
    def __init__(self, juzi):
        self.juzi = juzi
        self.base_url = "https://fanyi.baidu.com/v2transapi?from=en&to=zh"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0'}
        self.data = {
            'query': self.juzi,
            'from': 'en',
            'to': 'ch',
            'transtype': 'translang',
            'simple_means_flag': 3}

    def get_data(self):
        response = requests.post(self.base_url, headers=self.headers, data=self.data)
        return response.content.decode()

    def parse_data(self, data):
        # 将json字符串换成python字典
        dict_data = json.loads(data)

        # 使用键提取翻译结果
        result = dict_data['trans_result']['data'][0]['dst']  # ['dst']
        return result

    def run(self):
        # 构建url
        # 构建headers
        # 构建post参数
        # 发送post请求，获取响应
        data = self.get_data()
        # 解析响应
        result = self.parse_data(data)
        return result


if __name__ == '__main__':
    trans = Trans('how are you')
    data = trans.run()
    print(data)
