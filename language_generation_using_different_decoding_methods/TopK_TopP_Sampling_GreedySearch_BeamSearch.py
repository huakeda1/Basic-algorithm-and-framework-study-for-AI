#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict
import copy
np.random.seed=2021


# In[2]:


def temp_scaled_softmax(data,temp=1.0):
    prob=np.exp(data/temp)/np.sum(np.exp(data/temp),axis=-1)
    return prob
def greedy_search(prob):
    return np.argmax(prob)
    
def topk_sampling(prob,k=10,verbose=True):
    # this function is only used to process one single example
    if prob.ndim>1:
        prob=prob[0]
    topk_label=np.argsort(prob)[-k:]
    topk_prob=prob[topk_label]/np.sum(prob[topk_label])
    label=np.random.choice(topk_label,p=topk_prob)
    if verbose:
        print('orig_all_prob:{}'.format(prob))
        print('**********************')
        print('topk_sorted_label:{}'.format(topk_label))
        print('topk_sorted_prob:{}'.format(prob[topk_label]))
        print('**********************')
        print('topk_new_prob:{}'.format(topk_prob))
        print('finally sampled label:{}'.format(label))
    return label
def topp_sampling(prob,p=0.9,verbose=True):
    # this function is only used to process one single example
    if prob.ndim>1:
        prob=prob[0]
    sorted_htol_label=np.argsort(prob)[::-1]
    sorted_htol_prob=prob[sorted_htol_label]
    for i in range(prob.shape[0]):
        if np.sum(sorted_htol_prob[:i+1])>=p:
            break
    topp_htol_label=sorted_htol_label[:i+1]
    topp_htol_prob=sorted_htol_prob[topp_htol_label]/np.sum(sorted_htol_prob[topp_htol_label])
    label=np.random.choice(topp_htol_label,p=topp_htol_prob)
    if verbose:
        print('orig_all_prob:{}'.format(prob))
        print('**********************')
        print('topp_sorted_label:{}'.format(topp_htol_label))
        print('topp_sorted_prob:{}'.format(sorted_htol_prob[topp_htol_label]))
        print('**********************')
        print('topp_new_prob:{}'.format(topp_htol_prob))
        print('finally sampled label:{}'.format(label))


# In[3]:


test_data=np.array([[3,2,4,1]]).astype(float)
print(temp_scaled_softmax(test_data,temp=1.0))
print(temp_scaled_softmax(test_data,temp=0.8))
print(temp_scaled_softmax(test_data,temp=0.6))
print(temp_scaled_softmax(test_data,temp=0.4))
print(temp_scaled_softmax(test_data,temp=0.2))
print(temp_scaled_softmax(test_data,temp=0.1))
print(temp_scaled_softmax(test_data,temp=0.01))


# In[4]:


test_data=np.array([[3,2,4,1]]).astype(float)
prob=temp_scaled_softmax(test_data)
topk_label=topk_sampling(prob,k=3)
print('**********************')
topp_label=topp_sampling(prob,p=0.9)


# In[5]:


class beam_structure(object):
    def __init__(self,decode_start_input=0):
        self.storage=[(decode_start_input,1.0)]
    def add_item(self,label,prob):
        self.storage.append((label,prob))
    def get_total_prob(self,):
        labels,probs=zip(*self.storage)
        return np.exp(np.sum(np.log(probs)))
    def get_all_labels(self,):
        labels,probs=zip(*self.storage)
        return labels
    def get_all_probs(self,):
        labels,probs=zip(*self.storage)
        return probs
class assume_model(object):
    def __init__(self,label_dim=5):
        self.label_dim=label_dim
    def __call__(self,inputs):
        return np.random.randn(self.label_dim)
def beam_search(model,beam_num=3,max_num=10,softmax_temp=1.0):
    index=0
    total_beams=[beam_structure() for _ in range(beam_num)]
    while index<max_num:
        all_current_beams=[]
        for i in range(beam_num):
            inputs=total_beams[i].get_all_labels()
            outputs=model(inputs)
            prob=temp_scaled_softmax(outputs,softmax_temp)
            topk_labels=np.argsort(prob)[-beam_num:]
            topk_probs=prob[topk_labels]
            for label,prob in zip(topk_labels,topk_probs):
                new_beam=copy.deepcopy(total_beams[0])
                new_beam.add_item(label,prob)
                all_current_beams.append(new_beam)
            if index==0:
                break
        exist_data=defaultdict(list)
        filtered_beams=[]
        for current_beam in all_current_beams:
            label,prob=current_beam.storage[-1]
            if label not in exist_data:
                exist_data[label].append(current_beam)
        all_current_beams=[]
        for label,current_beams in exist_data.items():
            all_current_beams.append(sorted(current_beams,key=lambda x:x.storage[-1][-1],reverse=True)[0])         
        total_beams=sorted(all_current_beams,key=lambda x:x.get_total_prob(),reverse=True)[:beam_num]
        for i in range(beam_num):
            print('step:{},label_prob:{},accumulate_prob:{}'.format(index+1,total_beams[i].storage,total_beams[i].get_total_prob()))
        index+=1
    optimized_beam=sorted(total_beams,key=lambda x:x.get_total_prob(),reverse=True)[0]
    return optimized_beam.get_all_labels()   


# In[6]:


model=assume_model(label_dim=5)
beam_search(model,beam_num=3,max_num=5,softmax_temp=0.7)


# In[ ]:




