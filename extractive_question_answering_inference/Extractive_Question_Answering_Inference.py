#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForQuestionAnswering,AutoTokenizer
import torch


# In[2]:


model=AutoModelForQuestionAnswering.from_pretrained('./pretrained_model',return_dict=True)
tokenizer=AutoTokenizer.from_pretrained('./pretrained_model')


# In[3]:


text=r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""


# In[4]:


questions = ["How many pretrained models are available in 🤗 Transformers?","What does 🤗 Transformers provide?","🤗 Transformers provides interoperability between which frameworks?"]


# In[5]:


import torch.nn.functional as F
for question in questions:
    inputs=tokenizer(question,text,add_special_tokens=True,return_tensors='pt')
    input_ids=inputs['input_ids'].tolist()[0]
    text_tokens=tokenizer.convert_ids_to_tokens(input_ids)
    result=model(**inputs)
    answer_start_scores,answer_end_scores=result.start_logits,result.end_logits
    
    answer_start_greed=torch.argmax(answer_start_scores,dim=-1)
    answer_end_greed=torch.argmax(answer_end_scores,dim=-1)+1
    # here and false is only used to test the abnormal situations that the prediction of end index is smaller than that of start index
    if answer_start_greed<=answer_end_greed and False:
        answer_start=answer_start_greed
        answer_end=answer_end_greed
    else:
        q_len=len(tokenizer.encode(question,add_special_tokens=True))
        answer_start_probs=torch.softmax(answer_start_scores,dim=-1)[0,q_len:-1]
        answer_end_probs=torch.softmax(answer_end_scores,dim=-1)[0,q_len:-1]
        start_end,score=None,-1
        max_a_len=20
        for start,start_p in enumerate(answer_start_probs):
            for end,end_p in enumerate(answer_end_probs):
                if end>=start and end<start+max_a_len:
                    if start_p*end_p>score:
                        score=start_p*end_p
                        start_end=(start,end)
        relative_start,relative_end=start_end
        answer_start=relative_start+q_len
        answer_end=relative_end+q_len+1
    answer=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f'Text:{text_tokens}')
    print(f'Question:{question}')
    print(f'Start:{answer_start},End:{answer_end}')
    print(f'Answer:{answer}')


# In[6]:


import torch.nn.functional as F
for question in questions:
    inputs=tokenizer(question,text,add_special_tokens=True,return_tensors='pt')
    input_ids=inputs['input_ids'].tolist()[0]
    text_tokens=tokenizer.convert_ids_to_tokens(input_ids)
    result=model(**inputs)
    answer_start_scores,answer_end_scores=result.start_logits,result.end_logits
    
    answer_start_greed=torch.argmax(answer_start_scores,dim=-1)
    answer_end_greed=torch.argmax(answer_end_scores,dim=-1)+1
    if answer_start_greed<=answer_end_greed:
        answer_start=answer_start_greed
        answer_end=answer_end_greed
    else:
        q_len=len(tokenizer.encode(question,add_special_tokens=True))
        answer_start_probs=torch.softmax(answer_start_scores,dim=-1)[0,q_len:-1]
        answer_end_probs=torch.softmax(answer_end_scores,dim=-1)[0,q_len:-1]
        start_end,score=None,-1
        max_a_len=20
        for start,start_p in enumerate(answer_start_probs):
            for end,end_p in enumerate(answer_end_probs):
                if end>=start and end<start+max_a_len:
                    if start_p*end_p>score:
                        score=start_p*end_p
                        start_end=(start,end)
        answer_start,answer_end=start_end
        answer_end+=1
    answer=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f'Text:{text_tokens}')
    print(f'Question:{question}')
    print(f'Start:{answer_start},End:{answer_end}')
    print(f'Answer:{answer}')


# In[7]:


answer_end_scores.shape


# In[8]:


len(input_ids)


# # Extractive Question Answering(Inference)
# Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the **SQuAD** dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the **run_squad.py** and **run_tf_squad.py** scripts.
# 
# ## Packages
# - Transformers 3.5.0
# - Torch
# 
# ## The process is the following:
# - 1) Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it with the weights stored in the checkpoint.
# 
# - 2) Define a text and a few questions.
# 
# - 3) Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators token type ids and attention masks.
# 
# - 4) Pass this sequence through the model. This outputs a range of scores across the entire sequence tokens (question and text), for both the start and end positions.
# 
# - 5) Compute the softmax of the result to get probabilities over the tokens.
# 
# - 6) Fetch the tokens from the identified start and stop values, convert those tokens to a string.
# 
# - 7) Print the results.
# 
# ## Pretrained model
# You can download the pretrained weights from the [link](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main)
# 
# ## Special code
# ```python
# for question in questions:
#     inputs=tokenizer(question,text,add_special_tokens=True,return_tensors='pt')
#     input_ids=inputs['input_ids'].tolist()[0]
#     text_tokens=tokenizer.convert_ids_to_tokens(input_ids)
#     result=model(**inputs)
#     answer_start_scores,answer_end_scores=result.start_logits,result.end_logits
#     
#     answer_start_greed=torch.argmax(answer_start_scores,dim=-1)
#     answer_end_greed=torch.argmax(answer_end_scores,dim=-1)
# 
#     if answer_start_greed<=answer_end_greed:
#         answer_start=answer_start_greed
#         answer_end=answer_end_greed+1
#     else:
#         q_len=len(tokenizer.encode(question,add_special_tokens=True))
#         answer_start_probs=torch.softmax(answer_start_scores,dim=-1)[0,q_len:-1]
#         answer_end_probs=torch.softmax(answer_end_scores,dim=-1)[0,q_len:-1]
#         start_end,score=None,-1
#         max_a_len=20
#         for start,start_p in enumerate(answer_start_probs):
#             for end,end_p in enumerate(answer_end_probs):
#                 if end>=start and end<start+max_a_len:
#                     if start_p*end_p>score:
#                         score=start_p*end_p
#                         start_end=(start,end)
#         relative_start,relative_end=start_end
#         answer_start=relative_start+q_len
#         answer_end=relative_end+q_len+1
#     answer=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
# ```

# In[ ]:




