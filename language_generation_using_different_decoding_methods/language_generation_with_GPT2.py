#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from transformers import TFGPT2LMHeadModel,GPT2Tokenizer


# In[2]:


tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
model=TFGPT2LMHeadModel.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id)


# In[3]:


print('bos_token_id:{}'.format(tokenizer.bos_token_id))
print('eos_token_id:{}'.format(tokenizer.eos_token_id))
print('pad_token_id:{}'.format(tokenizer.pad_token_id))
print('unk_token_id:{}'.format(tokenizer.unk_token_id))
print('cls_token_id:{}'.format(tokenizer.cls_token_id))
print('sep_token_id:{}'.format(tokenizer.sep_token_id))
print('mask_token_id:{}'.format(tokenizer.mask_token_id))


# In[4]:


input_ids=tokenizer.encode('I enjoy walking with my cute dog',return_tensors='tf')
print(input_ids)
greedy_output=model.generate(input_ids,max_length=50,)
print(greedy_output)
print('Output:\n'+100*'-')
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))


# In[5]:


# activate beam search and early_stopping
beam_output=model.generate(input_ids,max_length=50,num_beams=5,early_stopping=True)
print('Output:\n'+100*'-')
print(tokenizer.decode(beam_output[0],skip_special_tokens=True))


# In[6]:


# set no_repeat_ngram_size to 2
beam_outputs=model.generate(input_ids,max_length=50,num_beams=5,no_repeat_ngram_size=2,num_return_sequences=5,early_stopping=True)
print('Output:\n'+100*'-')
for i,beam_output in enumerate(beam_outputs):
    print("{}: {}".format(i,tokenizer.decode(beam_output,skip_special_tokens=True)))


# In[7]:


tf.random.set_seed(2021)
beam_sample_outputs=model.generate(input_ids,max_length=50,num_beams=5,no_repeat_ngram_size=5,do_sample=True,top_k=30,temperature=0.7,top_p=0.95,num_return_sequences=5,early_stopping=True)
print('Output:\n'+100*'-')
for i,beam_sample_output in enumerate(beam_sample_outputs):
    print("{}: {}".format(i,tokenizer.decode(beam_sample_output,skip_special_tokens=True)))


# In[ ]:




