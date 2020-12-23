#!/usr/bin/env python
# coding: utf-8

# # Using built-in function from transformers to create the dataloader for squad

# In[1]:


from transformers.data.processors.squad import SquadV2Processor
from transformers import squad_convert_examples_to_features
from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import multiprocessing
import time


# In[2]:


# Loading a V2 processor
squad_v2_data_dir='./squad_v2'
bert_dir="./pretrained_model/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_dir)
processor = SquadV2Processor()
train_examples = processor.get_train_examples(squad_v2_data_dir,'train-v2.0.json')

# Loading a V1 processor
# processor = SquadV1Processor()
# examples = processor.get_dev_examples(squad_v1_data_dir)


# In[3]:


# the following code can be used to query the internal element of one class
dir(train_examples[0])


# In[4]:


len(train_examples)


# In[5]:


from tqdm import tqdm
impossible_counts=0
display=False
for i,example in enumerate(tqdm(train_examples)):
    if train_examples[i].is_impossible:
        print(train_examples[i].qas_id)
        print(train_examples[i].question_text)
        print(train_examples[i].context_text)
        print(train_examples[i].answer_text)
        print(train_examples[i].title)
        print(train_examples[i].is_impossible)
        print(train_examples[i].answers)
        print(train_examples[i].start_position)
        print(train_examples[i].end_position)
        impossible_counts+=1
    else:
        if display==False:
            print(train_examples[i].qas_id)
            print(train_examples[i].question_text)
            print(train_examples[i].context_text)
            print(train_examples[i].answer_text)
            print(train_examples[i].title)
            print(train_examples[i].is_impossible)
            print(train_examples[i].answers)
            print(train_examples[i].start_position)
            print(train_examples[i].end_position)
        display=True
    if impossible_counts==1:break


# In[6]:


train_features, train_dataset = squad_convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=True,
            return_dataset="pt",
            threads=multiprocessing.cpu_count(),
        )


# In[7]:


# the following code can be used to query the internal element of one class
# dir(train_features[0])


# In[8]:


train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)


# In[9]:


start_time=time.time()
device='cuda:0' if torch.cuda.is_available() else 'cpu'
for step, batch in enumerate(tqdm(train_dataloader)):
    batch = tuple(t.to(device) for t in batch)
#     inputs = {
#         "input_ids": batch[0],
#         "attention_mask": batch[1],
#         "token_type_ids": batch[2],
#         "start_positions": batch[3],
#         "end_positions": batch[4],
#     }
#     print('start_positions:',batch[3])
#     print('end_positions:',batch[4])
print('Time cost for getting all batches:',time.time()-start_time)


# # Building a new dataset inherited from torch.utils.data.Dataset

# In[10]:


import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import json
from tqdm import tqdm_notebook
import multiprocessing
from tqdm import tqdm


# In[11]:


class SquadDataset(Dataset):
    def __init__(self,data_path,vocab_path,max_length=384):
        super(SquadDataset,self).__init__()
        self.data_path=data_path
        self.data_samples,(self.data_questionids,self.data_questions,self.data_contexts,self.data_answers)=self.get_samples(data_path)
        self.tokenizer=AutoTokenizer.from_pretrained(vocab_path)
        self.max_length=max_length
    def get_samples(self,data_path):
        data_contexts,data_questions,data_questionids,data_answers=[],[],[],[]
        data_samples=[]
        with open(self.data_path,'r',encoding='utf-8') as f:
            dataset=json.load(f)
        for _,data in tqdm_notebook(enumerate(dataset['data'])):
            for _,paragraph in tqdm_notebook(enumerate(data['paragraphs'])):
                context=paragraph['context']
                if context not in data_contexts:
                    data_contexts.append(context)
                for qa in paragraph['qas']:
                    qid=qa['id']
                    if qid not in data_questionids:
                        data_questionids.append(qid)
                    question=qa['question']
                    if question not in data_questions:
                        data_questions.append(question)
                    current_answer=[]
                    for answer in qa['answers']:
                        text=answer['text']
                        if (len(text)==0) or (text in current_answer):
                            continue
                        else:
                            current_answer.append(text)
                        answer=answer['answer_start'] 
                        if text not in data_answers:
                            data_answers.append(text)
                        data_samples.append((data_questionids.index(qid),data_questions.index(question),data_contexts.index(context),data_answers.index(text)))
        return data_samples,(data_questionids,data_questions,data_contexts,data_answers)
    def get_answer_start(self,answer_ids,sequence_ids):
        for i in range(len(sequence_ids)):
            if sequence_ids[i:i+len(answer_ids)]==answer_ids:
                return i
        return -1
    def __getitem__(self,i):
        qid_index,question_index,context_index,answer_idnex=self.data_samples[i]
        qid,question,context,answer=self.data_questionids[qid_index],self.data_questions[question_index],self.data_contexts[context_index],self.data_answers[answer_idnex]
        output=self.tokenizer.encode_plus(question,context,add_special_tokens=True,max_length=self.max_length,truncation=True)
        pure_answer_ids=self.tokenizer.encode(answer)[1:-1]
        start=self.get_answer_start(pure_answer_ids,output['input_ids'])
        if start!=-1:
            label=(start,start+len(pure_answer_ids)-1)
        else:
            label=(0,0)
        return output,label
    def __len__(self):
        return len(self.data_samples)
def collate_func(batch):
    def padding(indices,max_length,pad_idx=0):
        pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
        return torch.tensor(pad_indices)
#     input_ids=[output['input_ids'] for output,label in batch]
#     max_length=max([len(t) for t in input_ids])
#     labels=torch.tensor([label for output,label in batch])
#     token_type_ids=[output['token_type_ids'] for output,label in batch]
#     attention_mask=[output['attention_mask'] for output,label in batch]
    
    result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
    input_ids,token_type_ids,attention_mask,labels=zip(*result)
    labels=torch.tensor(labels)
    max_length=max([len(t) for t in input_ids])
    
    input_ids_padded=padding(input_ids,max_length)
    token_type_ids_padded=padding(token_type_ids,max_length)
    attention_mask_padded=padding(attention_mask,max_length)
    return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels


# In[12]:


import time
start_time=time.time()
data_path='./squad_v2/train-v2.0.json'
vocab_path="./pretrained_model/bert-base-uncased"
squad_dataset=SquadDataset(data_path=data_path,vocab_path=vocab_path)
print('building_time:',time.time()-start_time)

squad_dataloader=DataLoader(squad_dataset,batch_size=16,collate_fn=collate_func,num_workers=multiprocessing.cpu_count())

start_time=time.time()
device='cuda:0' if torch.cuda.is_available() else 'cpu'
for i,batch in tqdm(enumerate(squad_dataloader)):
    batch = tuple(t.to(device) for t in batch)
    input_ids_padded,token_type_ids_padded,attention_mask_padded,labels=batch[0],batch[1],batch[2],batch[3]
print('cost_time:',time.time()-start_time)


# In[13]:


from multiprocessing import Pool
import multiprocessing
import time
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
class SquadDataset_new(Dataset):
    def __init__(self,data_path,vocab_path,max_length=384):
        super(SquadDataset_new,self).__init__()
        self.data_path=data_path
        self.data_samples=self.get_samples(data_path)
        self.tokenizer=BertTokenizer.from_pretrained(vocab_path)
        self.max_length=max_length
    def get_examples_from_data(self,data):
        examples=[]
        title=data['title']
        for paragraph in data['paragraphs']:
            context=paragraph['context']
            for qa in paragraph['qas']:
                qid=qa['id']
                question=qa['question']
                is_impossible=qa['is_impossible']
                if is_impossible:
                    current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':''}
                    examples.append(current_example)
                else:
                    for answer in qa['answers']:
                        text=answer['text']    
                        answer=answer['answer_start']
                        current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':text}
                        examples.append(current_example)
        return examples
    def get_samples(self,data_path):
        with open(data_path,'r',encoding='utf-8') as f:
            dataset=json.load(f)
            all_data=dataset['data']
        processor=multiprocessing.cpu_count()
        p=Pool(processor)
        result_list=p.map(self.get_examples_from_data,all_data)
        all_examples=[]
        for examples in result_list:
            all_examples+=examples
        p.close()
        return all_examples
    def get_answer_start(self,answer_ids,sequence_ids):
        for i in range(len(sequence_ids)):
            if sequence_ids[i:i+len(answer_ids)]==answer_ids:
                return i
        return -1
    def __getitem__(self,i):
        
        qid=self.data_samples[i]['qas_id']
        question=self.data_samples[i]['question_text']
        context=self.data_samples[i]['context_text']
        answer=self.data_samples[i]['answer_text']
        output=self.tokenizer.encode_plus(question,context,add_special_tokens=True,max_length=self.max_length,truncation=True)
        pure_answer_ids=self.tokenizer.encode(answer)[1:-1]
        start=self.get_answer_start(pure_answer_ids,output['input_ids'])
        if start!=-1:
            label=(start,start+len(pure_answer_ids)-1)
        else:
            label=(0,0)
        return output,label
    def __len__(self):
        return len(self.data_samples)
def collate_func(batch):
    def padding(indices,max_length,pad_idx=0):
        pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
        return torch.tensor(pad_indices)
#     input_ids=[output['input_ids'] for output,label in batch]
#     max_length=max([len(t) for t in input_ids])
#     labels=torch.tensor([label for output,label in batch])
#     token_type_ids=[output['token_type_ids'] for output,label in batch]
#     attention_mask=[output['attention_mask'] for output,label in batch]
    
    result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
    input_ids,token_type_ids,attention_mask,labels=zip(*result)
    labels=torch.tensor(labels)
    max_length=max([len(t) for t in input_ids])
    
    input_ids_padded=padding(input_ids,max_length)
    token_type_ids_padded=padding(token_type_ids,max_length)
    attention_mask_padded=padding(attention_mask,max_length)
    return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels


# In[14]:


import time
start_time=time.time()
data_path='./squad_v2/train-v2.0.json'
vocab_path="./pretrained_model/bert-base-uncased"
squad_dataset=SquadDataset_new(data_path=data_path,vocab_path=vocab_path)
print('building_time:',time.time()-start_time)
processor=multiprocessing.cpu_count()
squad_dataloader=DataLoader(squad_dataset,batch_size=16,collate_fn=collate_func,num_workers=processor)

start_time=time.time()
device='cuda:0' if torch.cuda.is_available() else 'cpu'
for i,batch in tqdm(enumerate(squad_dataloader)):
    batch = tuple(t.to(device) for t in batch)
    input_ids_padded,token_type_ids_padded,attention_mask_padded,labels=batch[0],batch[1],batch[2],batch[3]
print('cost_time:',time.time()-start_time)


# Python多进程进行文件预处理
# https://blog.csdn.net/mingo220/article/details/105372025

# # Building a new dataset by python yield way

# In[15]:


from multiprocessing import Pool
import multiprocessing
import time
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import time
from tqdm import tqdm
import math

class SquadDataset2():
    def __init__(self,data_path,vocab_path,max_length=384,batch_size=16):
        self.data_path=data_path
        self.tokenizer=BertTokenizer.from_pretrained(vocab_path)
        self.data_samples=self.get_samples(data_path)
        self.max_length=max_length
        self.batch_size=batch_size
        self.total_samples=len(self.data_samples)
        self.total_batches=self.total_samples//self.batch_size if self.total_samples%self.batch_size==0 else self.total_samples//self.batch_size+1
    def get_examples_from_data(self,data):
        examples=[]
        title=data['title']
        for paragraph in data['paragraphs']:
            context=paragraph['context']
            for qa in paragraph['qas']:
                qid=qa['id']
                question=qa['question']
                is_impossible=qa['is_impossible']
                if is_impossible:
                    current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':''}
                    examples.append(current_example)
                else:
                    for answer in qa['answers']:
                        text=answer['text']    
                        answer=answer['answer_start']
                        current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':text}
                        examples.append(current_example)
        return examples
    def get_samples(self,data_path):
        with open(data_path,'r',encoding='utf-8') as f:
            dataset=json.load(f)
            all_data=dataset['data']
        processor=multiprocessing.cpu_count()
        p=Pool(processor//2)
        result_list=p.map(self.get_examples_from_data,all_data)
        all_examples=[]
        for examples in result_list:
            all_examples+=examples
        p.close()
        return all_examples
    def get_answer_start(self,answer_ids,sequence_ids):
        for i in range(len(sequence_ids)):
            if sequence_ids[i:i+len(answer_ids)]==answer_ids:
                return i
        return -1
    def get_answer_position(self,example):
        question,context,answer=example['question_text'],example['context_text'],example['answer_text']
        output=self.tokenizer.encode_plus(question,context,add_special_tokens=True,max_length=self.max_length,truncation=True)
        pure_answer_ids=self.tokenizer.encode(answer)[1:-1]
        start=self.get_answer_start(pure_answer_ids,output['input_ids'])
        if start!=-1:
            label=(start,start+len(pure_answer_ids)-1)
        else:
            label=(0,0)
        return output,label
    def get_output_from_example(self,example):
        def padding(indices,max_length,pad_idx=0):
            pad_indices=indices+[pad_idx]*max(0,max_length-len(indices))
            return pad_indices
        question,context,text=example['question_text'],example['context_text'],example['answer_text']
        output,label=self.get_answer_position(question,context,text)
        input_ids,token_type_ids,attention_mask=output['input_ids'],output['token_type_ids'],output['attention_mask']

        input_ids_padded=padding(input_ids,self.max_length)
        token_type_ids_padded=padding(token_type_ids,self.max_length)
        attention_mask_padded=padding(attention_mask,self.max_length)
        return input_ids_padded,token_type_ids_padded,attention_mask_padded,label
#     def process_func(data, index, size):  # data 传入数据，index 数据分片索引，size进程数
#         size = math.ceil(len(data) / size)
#         start = size * index
#         end = (index + 1) * size if (index + 1) * size < len(data) else len(data)
#         temp_data = data[start:end]
        
#         return get_new_examples(temp_data)
    def collate_func(self,batch):
        def padding(indices,max_length,pad_idx=0):
            pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
            return pad_indices
    #     input_ids=[output['input_ids'] for output,label in batch]
    #     max_length=max([len(t) for t in input_ids])
    #     labels=torch.tensor([label for output,label in batch])
    #     token_type_ids=[output['token_type_ids'] for output,label in batch]
    #     attention_mask=[output['attention_mask'] for output,label in batch]

        result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for (output,label) in batch]
        input_ids,token_type_ids,attention_mask,labels=zip(*result)
        labels=labels
        max_length=max([len(t) for t in input_ids])

        input_ids_padded=padding(input_ids,max_length)
        token_type_ids_padded=padding(token_type_ids,max_length)
        attention_mask_padded=padding(attention_mask,max_length)
        return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels
    def get_batched_data(self):
        for index in tqdm(range(self.total_batches)):
            start = self.batch_size * index
            end = (index + 1) * self.batch_size if (index + 1) * self.batch_size < self.total_samples else self.total_samples
            temp_data = self.data_samples[start:end]
            # processor=multiprocessing.cpu_count()
            # p=Pool(processor)
            batch=map(self.get_answer_position,temp_data)
            # p.close()
            input_ids_batched,token_type_ids_batched,attention_mask_batched,labels_batched=self.collate_func(batch)
            yield torch.tensor(input_ids_batched),torch.tensor(token_type_ids_batched),torch.tensor(attention_mask_batched),torch.tensor(labels_batched)


# In[16]:


import time
start_time=time.time()
data_path='./squad_v2/train-v2.0.json'
vocab_path="./pretrained_model/bert-base-uncased"
squad_dataset=SquadDataset2(data_path=data_path,vocab_path=vocab_path)
print('building_time:',time.time()-start_time)

start_time=time.time()
device='cuda:0' if torch.cuda.is_available() else 'cpu'
for i,batch in tqdm(enumerate(squad_dataset.get_batched_data())):
    batch = tuple(t.to(device) for t in batch)
    if i==0:
        print(batch[0].shape,batch[-1].shape)
# (input_ids_padded,token_type_ids_padded,attention_mask_padded,labels)
print('cost_time:',time.time()-start_time)


# # Another way of multiprocessing data

# In[17]:


def get_new_examples(part_datas):
    examples=[]
    for data in part_datas:
        title=data['title']
        for paragraph in data['paragraphs']:
            context=paragraph['context']
            for qa in paragraph['qas']:
                qid=qa['id']
                question=qa['question']
                is_impossible=qa['is_impossible']
                if is_impossible:
                    current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':''}
                    examples.append(current_example)
                else:
                    for answer in qa['answers']:
                        text=answer['text']    
                        answer=answer['answer_start']
                        current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':text}
                        examples.append(current_example)
    return examples


# In[18]:


from multiprocessing import Pool
import multiprocessing
import math
import time
start_time=time.time()
data_path='./squad_v2/train-v2.0.json'
with open(data_path,'r',encoding='utf-8') as f:
    dataset=json.load(f)
    all_data=dataset['data']
def process_func(data, index, size):  # data 传入数据，index 数据分片索引，size进程数
    size = math.ceil(len(data) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(data) else len(data)
    temp_data = data[start:end]
    return get_new_examples(temp_data)
processor=multiprocessing.cpu_count()
p=Pool(processor)
all_examples=[]
for i in tqdm(range(processor)):
    all_examples+=p.apply_async(process_func,args=(all_data,i,processor,)).get()
p.close()
p.join()
print('time cost:',time.time()-start_time)


# # Different way of building dataset for squad data.
# We study different ways of building dataset for squad data, these ways can also be used in other relevant application, we also make full use of the multiprocessor to process the data, this way of doing can really increase the efficiency of preprocessing.
# 
# 
# ## Packages
# - Torch
# - multiprocessing
# - math
# - json
# - transformers
# - numpy
# 
# ## The following ways of building dataset for squad data are studied in this repository:
# - 1) Using built-in function from transformers to create the dataloader for squad
# 
# - 2) Building a new dataset inherited from torch.utils.data.Dataset.
# 
# - 3) Building a new dataset by python yield way.
# 
# - 4) Another way of multiprocessing data.
# 
# 
# ## Special code
# ```python
# # Using built-in function from transformers to create the dataset for squad
# tokenizer = BertTokenizer.from_pretrained(bert_dir)
# processor = SquadV2Processor()
# train_examples = processor.get_train_examples(squad_v2_data_dir,'train-v2.0.json')
# # the following code can be used to query the internal element of one class
# dir(train_examples[0])
# train_features, train_dataset = squad_convert_examples_to_features(
#             examples=train_examples,
#             tokenizer=tokenizer,
#             max_seq_length=384,
#             doc_stride=128,
#             max_query_length=64,
#             is_training=True,
#             return_dataset="pt",
#             threads=multiprocessing.cpu_count(),
#         )
# # Using multiprocessing to process the squad data
# from multiprocessing import Pool
# def get_samples(self,data_path):
#     with open(data_path,'r',encoding='utf-8') as f:
#         dataset=json.load(f)
#         all_data=dataset['data']
#     processor=multiprocessing.cpu_count()
#     p=Pool(processor)
#     result_list=p.map(self.get_examples_from_data,all_data)
#     all_examples=[]
#     for examples in result_list:
#         all_examples+=examples
#     p.close()
#     return all_examples
# # The core code of building a new dataset by python yield way.
#  def collate_func(self,batch):
#         def padding(indices,max_length,pad_idx=0):
#             pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
#             return pad_indices
#     #     input_ids=[output['input_ids'] for output,label in batch]
#     #     max_length=max([len(t) for t in input_ids])
#     #     labels=torch.tensor([label for output,label in batch])
#     #     token_type_ids=[output['token_type_ids'] for output,label in batch]
#     #     attention_mask=[output['attention_mask'] for output,label in batch]
# 
#         result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for (output,label) in batch]
#         input_ids,token_type_ids,attention_mask,labels=zip(*result)
#         labels=labels
#         max_length=max([len(t) for t in input_ids])
# 
#         input_ids_padded=padding(input_ids,max_length)
#         token_type_ids_padded=padding(token_type_ids,max_length)
#         attention_mask_padded=padding(attention_mask,max_length)
#         return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels
#     def get_batched_data(self):
#         for index in tqdm(range(self.total_batches)):
#             start = self.batch_size * index
#             end = (index + 1) * self.batch_size if (index + 1) * self.batch_size < self.total_samples else self.total_samples
#             temp_data = self.data_samples[start:end]
#             # processor=multiprocessing.cpu_count()
#             # p=Pool(processor)
#             batch=map(self.get_answer_position,temp_data)
#             # p.close()
#             input_ids_batched,token_type_ids_batched,attention_mask_batched,labels_batched=self.collate_func(batch)
#             yield torch.tensor(input_ids_batched),torch.tensor(token_type_ids_batched),torch.tensor(attention_mask_batched),torch.tensor(labels_batched)
#             
# # Another way of multiprocessing data.
# with open(data_path,'r',encoding='utf-8') as f:
#     dataset=json.load(f)
#     all_data=dataset['data']
# def process_func(data, index, size):  # data 传入数据，index 数据分片索引，size进程数
#     size = math.ceil(len(data) / size)
#     start = size * index
#     end = (index + 1) * size if (index + 1) * size < len(data) else len(data)
#     temp_data = data[start:end]
#     return get_new_examples(temp_data)
# processor=multiprocessing.cpu_count()
# p=Pool(processor)
# all_examples=[]
# for i in tqdm(range(processor)):
#     all_examples+=p.apply_async(process_func,args=(all_data,i,processor,)).get()
# p.close()
# p.join()
# ```

# In[ ]:




