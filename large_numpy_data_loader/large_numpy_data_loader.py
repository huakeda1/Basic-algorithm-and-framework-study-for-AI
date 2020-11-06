#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import tqdm
import os
import time
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def create_data(file,c_len=150,q_len=50,embedding_size=768):
    total_samples = 0
    if os.path.exists(file):
        print('data already stored in file')
        with open(file,'rb') as f:
            for index in range(1000000):
                try:
                    data=pickle.load(f)
                except EOFError:
                    break
                else:
                    total_samples += 1
        return total_samples
    with open(file,'wb') as f:
        for index in tqdm.tqdm_notebook(range(50000)):
            c_embed=np.random.randn(c_len,embedding_size)
            q_embed=np.random.randn(q_len,embedding_size)
            b=np.random.randint(c_len)
            e=np.random.randint(b,c_len)
            be=np.array([b,e],dtype=np.int32)
            output=(c_embed,q_embed,be)
            pickle.dump(output,f)
            total_samples += 1
    return total_samples

def get_loader(file,batch_size):
    inputs = []
    with open(file,'rb') as f:
        for index in range(1000000):
            try:
                data=pickle.load(f)
            except EOFError:
                break
            else:
                inputs.append(data)
                if len(inputs)==batch_size:
                    c_embeds,q_embeds,be=zip(*inputs)
                    output_c_embeds=np.stack(c_embeds,axis=0)
                    output_q_embeds=np.stack(q_embeds,axis=0)
                    output_be=np.stack(be,axis=0)
                    inputs=[]
                    yield ((output_c_embeds,output_q_embeds),output_be)
        if len(inputs)!=0:
            c_embeds,q_embeds,be=zip(*inputs)
            output_c_embeds=np.stack(c_embeds,axis=0)
            output_q_embeds=np.stack(q_embeds,axis=0)
            output_be=np.stack(be,axis=0)
            inputs=[]
            yield ((output_c_embeds,output_q_embeds),output_be)


# In[ ]:


# create and store data
file='large_context_question_data.pkl'
start_time=time.time()
total_samples=create_data(file,c_len=150,q_len=50,embedding_size=768)
print('total_samples:',total_samples)
print('It cost {} s to create and store data'.format(time.time()-start_time))


# In[4]:


# load and show data
file='large_context_question_data.pkl'
batch_size=32
total_batches = 0
second_time=time.time()
for (output_c_embeds,output_q_embeds),output_be in tqdm.tqdm_notebook(get_loader(file,batch_size)):
    total_batches += 1
    print(output_c_embeds.shape,output_q_embeds.shape,output_be.shape)
print('It cost {} s to show the data from file'.format(time.time()-second_time))
print('total_batches:',total_batches)


# In[ ]:




