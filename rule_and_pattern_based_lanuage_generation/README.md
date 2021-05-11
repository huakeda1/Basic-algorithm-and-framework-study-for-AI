## Simple language generation model and rationality judgement model  
### **n-gram language evaluation model**  
The model can be used to decide which sentence is proper according to their probability calculated by the following formulas:  
$P(word^{t+1}|word^{t},...,word^{1})=P(word^{t+1}|word^{t},...,word^{t-n+2})≈\frac{count(word^{t+1},...word^{t-n+2})}{count(word^{t},...word^{t-n+2})}$  
A large number of corpus should be provided so as to get more accurate result.

### **Rule based language generation model**  
The model can generate sentence according to specific grammar designed in advance，the typical grammar and relevant generated sentence are shown as below: 

#### Typical grammar:  
host = """
host = 寒暄 报数 询问 业务相关 结尾   
报数 = 我是 数字 号 ,  
数字 = 单个数字 | 数字 单个数字   
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9   
寒暄 = 称谓 打招呼 | 打招呼  
称谓 = 人称 ,  
人称 = 先生 | 女士 | 小朋友  
打招呼 = 你好 | 您好   
询问 = 请问你要 | 您需要  
业务相关 = 玩玩 具体业务  
玩玩 = 耍一耍 | 玩一玩  
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博  
结尾 = 吗？"""  

#### Typical generated sentence:  
女士,你好我是52号,您需要玩一玩打猎吗？  
您好我是453号,请问你要玩一玩打牌吗？  
你好我是93号,请问你要玩一玩打牌吗？  
女士,您好我是71号,请问你要耍一耍打猎吗？  
女士,您好我是296号,请问你要耍一耍喝酒吗？  

#### Core-code:
```
def generate_sentence_from_grammar(grammar_dict:dict,target_text:str):
    if target_text not in grammar_dict:return target_text
    targets=random.choice(grammar_dict[target_text])
    return ''.join([generate_sentence_from_grammar(grammar_dict,target) for target in targets])
```
### **Pattern match based language generation model**  
The model can generate sentence according to specific grammar designed in advance，the typical grammar and relevant generated sentence are shown as below: 

#### Typical pattern and generated response
```
'?*x想要?*y': ['?x想问你，你觉得?y有什么意义呢?', '为什么你想?y', '?x觉得... 你可以想想你很快就可以有?y了', '你看?x像?y不', '我看你就像?y']
```
```
input='我想要礼物'
output=['我想问你，你觉得礼物有什么意义呢？','为什么你想礼物','我觉得...你可以想想你很快就可以有礼物了','你看我像礼物不','我看你就像礼物']
```
#### Core-code:
```
def conclude_same(string1,string2):
    if string1=='' and string2=='':
        return True
    if conclude_pattern(string1[0]):
        return True
    if string1[0]==string2[0]:
        return conclude_same(string1[1:],string2[1:])
    return False

def get_same_start_point(rule_string,saying_string):
    if not rule_string:
        return len(saying_string)
    for index,char in enumerate(saying_string):
        if(char==rule_string[0]):
            if(conclude_same(rule_string[1:],saying_string[index+1:])):
                return index
    return None
def pattern_match_result(rule:list,saying:list):
    if not rule and not saying:
        return []
    string,rest=rule[0],rule[1:]
    if conclude_pattern(string):
        index=get_same_start_point(rest,saying)
        string=string.replace('?*','?')
        return [(string,saying[:index])]+pattern_match_result(rule[1:],saying[index:])
    if string==saying[0]:
        return pattern_match_result(rule[1:],saying[1:])
    else:
        print('not match')
def get_response_from_saying(saying:str,rule_responses:dict):
    language=check_lanuage(saying)
    rules=list(rule_responses.keys())
    saying_list=split_words(saying)
    max_nums=0
    max_rule=rules[0]
    best_result=[]
    for rule in rules:
        rule_list=split_words(rule)
        result=pattern_match(rule_list,saying_list)
        if len(result)>max_nums:
            max_nums=len(result)
            max_rule=rule
            best_result=result
    if best_result:
        pattern_result=match_result_to_dict(best_result)
        best_response=random.choice(rule_responses[max_rule])
        return ' '.join([pattern_result.get(word,word) for word in split_words(best_response)]) if not language else ''.join([pattern_result.get(word,word) for word in split_words(best_response)])
    print('{} can\'t be pattern matched to any pattern of answer mode designed in advance'.format(saying))
    return ''
```
