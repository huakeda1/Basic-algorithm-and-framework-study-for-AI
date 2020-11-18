#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers
print(transformers.__version__)


# In[ ]:


# pip install transformers==3.4.0


# In[2]:


from transformers import BertConfig,BertModel,BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[30]:


class Conv1D(nn.Module):
    def __init__(self,in_channels,out_channels,filter_sizes):
        super(Conv1D,self).__init__()
        self.convs=nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=fs) for fs in filter_sizes
        ])
        self.init_params()
    def init_params(self):
        for m in self.convs:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data,0.1)
    def forward(self,x):
        return [F.relu(conv(x)) for conv in self.convs]
class BertCNN(nn.Module):
    def __init__(self,config):
        super(BertCNN,self).__init__()
        self.num_labels=config.num_classes
        model_config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path,config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout=nn.Dropout(config.dropout)
        self.convs=Conv1D(config.hidden_size,config.num_filters,filter_sizes=config.filter_sizes)
        self.classifier=nn.Linear(len(config.filter_sizes)*config.num_filters,self.num_labels)
    def forward(self,x):
        """
        Args:
            input_ids: token_id
            token_type_ids: 0 means first sentence,1 means second sentence
            attention_mask: 1 means token, 0 means padding
        """
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        
        encoded_layer,_=self.bert(context,token_type_ids=token_type_ids,attention_mask=mask)
        # why the tensor should be permuted?
        encoded_layer=encoded_layer.permute(0,2,1)
        # conved[0] shape (batch_size,n_filters,-1)
        conved=self.convs(encoded_layer)
        # conved[0] shape (batch_size,n_filters)
        max_pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        # cat shape (batch_size,n_filters*len(filter_sizes))
        cat = self.dropout(torch.cat(max_pooled,dim=1))
        # logits shape (batch_size,num_labels)
        logits = self.classifier(cat)
        return logits
class Config():
    def __init__(self):
        self.bert_path='./home/aistudio/data/data56340'
        self.num_classes=13
        self.filter_sizes=[2,3,4]
        self.num_filters=256
        self.hidden_size=768
        self.dropout=0.1
config=Config()


# In[31]:


model=BertCNN(config)


# In[32]:


bert_params=[]
other_params=[]
for n,p in model.named_parameters():
    if 'bert' in n:
        bert_params.append((n,p))
    else:
        other_params.append((n,p))
bert_names,bert_parameters=zip(*bert_params)
other_names,other_parameters=zip(*other_params)
print(bert_names)
print(other_names)


# In[33]:


for n,p in model.bert.named_parameters():
    print(id(p))


# In[34]:


for n,p in model.named_parameters():
    print(id(p))


# In[18]:


bert_config=BertConfig.from_pretrained('./home/aistudio/data/data56340')


# In[19]:


bert_tokenizer=BertTokenizer.from_pretrained('./home/aistudio/data/data56340')


# In[20]:


bert_model=BertModel.from_pretrained('./home/aistudio/data/data56340')


# In[38]:


result=bert_tokenizer.encode_plus(text='今天天气真好，我应该考虑出去走走',add_special_tokens=True,max_length=128,truncation=True)
print(result)


# In[39]:


model((torch.tensor([result['input_ids']]),torch.tensor([result['token_type_ids']]),torch.tensor([result['attention_mask']])))


# In[40]:


from transformers import get_linear_schedule_with_warmup


# In[41]:


get_ipython().run_line_magic('pinfo', 'get_linear_schedule_with_warmup')


# In[42]:


# the following code is mainly used to unzip the required data to specific folder.
import os,zipfile
src_file='chinese_wobert_L-12_H-768_A-12.zip'
zf=zipfile.ZipFile(src_file)
zf.extractall('./home/aistudio/data/wobert')
zf.close


# In[43]:


get_ipython().system('ls ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12')


# In[66]:


# The following code is used to change the pretrained model from tf format to pytorch. 
get_ipython().run_line_magic('run', 'convert_bert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt   --bert_config_file ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_config.json   --pytorch_dump_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/pytorch_model.bin')


# In[68]:


wobert_dir="./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12"
wobert_config=BertConfig.from_pretrained(wobert_dir)
wobert_tokenizer=BertTokenizer.from_pretrained(wobert_dir)
wobert_model=BertModel.from_pretrained(wobert_dir,config=wobert_config)


# ## Transformer basic study
# We build a bert and cnn combined fine-tuned model on the base of the transformed pretrained bert model from Google, we know how to transform the pretrained model from the format of tf to pt, we learn how to load pretrained model in new defined network for downstream tasks, we understand the name and weights of layers so as to setup different optimizer values for different parameters of layers, the model we build here can used to do text classification.
# 
# ### Main Content
# - build a text classification model by incorporating pretrained model 
# - study name and weights of parameters
# - study the way of transforming pretrained weights from the format of tf to pt
# 
# ### Packages
# - torch
# - transformers
# - zipfile
# 
# ### Important functions
# - nn.Module
# - nn.Linear()
# - nn.parameter()
# - nn.ModuleList()
# - nn.Conv1d()
# - nn.init.xavier_normal_
# - nn.init.constant_
# - torch.permute()
# - torch.nn.functional.max_pool1d()
# - torch.cat()
# - nn.Dropout()
# 
# ### Special code
# ```python
# # layer parameters initialization 
# def init_params(self):
#         for m in self.convs:
#             nn.init.xavier_normal_(m.weight.data)
#             nn.init.constant_(m.bias.data,0.1)
# 
# # forward calculation
# def forward(self,x):
#         """
#         Args:
#             input_ids: token_id
#             token_type_ids: 0 means first sentence,1 means second sentence
#             attention_mask: 1 means token, 0 means padding
#         """
#         context = x[0]  # 输入的句子
#         mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         token_type_ids = x[2]
#         
#         encoded_layer,_=self.bert(context,token_type_ids=token_type_ids,attention_mask=mask)
#         # why the tensor should be permuted?
#         encoded_layer=encoded_layer.permute(0,2,1)
#         # conved[0] shape (batch_size,n_filters,-1)
#         conved=self.convs(encoded_layer)
#         # conved[0] shape (batch_size,n_filters)
#         max_pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
#         # cat shape (batch_size,n_filters*len(filter_sizes))
#         cat = self.dropout(torch.cat(max_pooled,dim=1))
#         # logits shape (batch_size,num_labels)
#         logits = self.classifier(cat)
#         return logits
# 
# # split none bert parameters from the model
# bert_params=[]
# other_params=[]
# for n,p in model.named_parameters():
#     if 'bert' in n:
#         bert_params.append((n,p))
#     else:
#         other_params.append((n,p))
# bert_names,bert_parameters=zip(*bert_params)
# other_names,other_parameters=zip(*other_params)
# print(bert_names)
# print(other_names)
# 
# # unzip the required data to specific folder.
# import os,zipfile
# src_file='chinese_wobert_L-12_H-768_A-12.zip'
# zf=zipfile.ZipFile(src_file)
# zf.extractall('./home/aistudio/data/wobert')
# zf.close
# 
# # transform pretrained weights from the format of tf to pt
# %run convert_bert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt \
#   --bert_config_file ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_config.json \
#   --pytorch_dump_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/pytorch_model.bin
# ```

# In[ ]:




