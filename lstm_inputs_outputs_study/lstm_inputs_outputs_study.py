#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
batch_size=5
seq_len=8
input_size=10
num_layers=3
bidirec=2
hidden_size=20
inputs=torch.randn(batch_size,seq_len,input_size)
h0=torch.randn(num_layers*bidirec,batch_size,hidden_size)
c0=torch.randn(num_layers*bidirec,batch_size,hidden_size)
model=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True if bidirec==2 else False)
output,(h_n,c_n)=model(inputs,(h0,c0))
print(output.shape)
print(h_n.shape)
print(c_n.shape)
# new_hidden can be used for downsteam tasks
new_hidden=torch.cat((h_n[-2,...],h_n[-1,...]),dim=1)
print(new_hidden.shape)
another_hidden=torch.cat((output[:,-1,:20],output[:,0,20:]),dim=1)
print(another_hidden.shape)
print(torch.eq(new_hidden,another_hidden))
new_h_n=h_n.view(num_layers,bidirec,batch_size,hidden_size)
print(new_h_n.shape)
next_hidden=torch.cat((new_h_n[-1,0,...],new_h_n[-1,1,...]),dim=1)
print(next_hidden.shape)
print(torch.eq(new_hidden,next_hidden))


# ## Lstm inputs and outputs study
# Build a simple program to get a clearly understanding about the inputs and outputs of **multi-layer bidirectional** LSTM. 
# 
# ### Packages
# - torch
# 
# ### Important functions
# - torch.randn()
# - torch.nn.LSTM()
# - torch.eq()
# - torch.cat()
# - torch.view()
# 
# ### Main Content
# 
# The dimensions of inputs and outputs of LSTM are depicted as below:  
# output,(h_n,c_n)=model(input,(h_0,c_0))  
# 
# inputs=(input,(h_0,c_0))  
# input.shape=(batch_size,seq_len,input_size)  
# h_0.shape=(num_layers*directions,batch_size,hidden_size)  
# c_0.shape=(num_layers*directions,batch_size,hidden_size)  
# 
# outputs=(output,(h_n,c_n))  
# output.shape=(batch_size,seq_len,2*hidden_size(when bidirectional=True))  
# h_n.shape=(num_layers*directions,batch_size,hidden_size)  
# c_n.shape=(num_layers*directions,batch_size,hidden_size)  
# 
# h_n[-1,batch_size,hidden_size] means the last backward hidden state of the last layer(when bidirectional=True)  
# h_n[-2,batch_size,hidden_size] means the last forward hidden state of the last layer(when bidirectional=True)  
# 
# h_n[-3,batch_size,hidden_size] means the last backward hidden state of the second last layer(when bidirectional=True)  
# h_n[-4,batch_size,hidden_size] means the last forward hidden state of the second last layer(when bidirectional=True)  
# 
# output[:,0,hidden_size:]=h_n[-1,batch_size,hidden_size]  
# output[:,-1,:hidden_size]=h_n[-2,batch_size,hidden_size]  
# 
# You can also review the [link](https://blog.csdn.net/qq_39777550/article/details/106659150) for further study.  

# In[ ]:




