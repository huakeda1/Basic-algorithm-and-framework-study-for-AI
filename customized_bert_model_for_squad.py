#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Pool
import multiprocessing
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from transformers import BertConfig,BertTokenizer,BertModel,AdamW
from tqdm import tqdm
import time
import argparse
import collections
import numpy as np
import os
import re
import string
import sys


# In[2]:


class SquadDataset(Dataset):
    def __init__(self,data_path,vocab_path,max_length=384):
        super(SquadDataset,self).__init__()
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
                if "is_impossible" in qa:
                    is_impossible=qa['is_impossible']
                else:
                    is_impossible=len(qa['answers'])==0
                if is_impossible:
                    current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':''}
                    examples.append(current_example)
                else:
                    current_answer_collectors=[]
                    for answer in qa['answers']:
                        text=answer['text']
                        if text in current_answer_collectors:
                            continue
                        else:
                            current_answer_collectors.append(text)
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
        output=self.tokenizer.encode_plus(question,context,add_special_tokens=True,max_length=self.max_length,padding='max_length',truncation=True)
        answer=self.data_samples[i]['answer_text']
        output.update({'answer_text':answer if answer!=None else '','qas_id':qid})
        if answer=='' or answer==None:
            label=(0,0)
        else:
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
    # if padding = 'max_length' is not used in tokenizer.encode_plust fuction, then you should pad the sequences to the same size.
    
#     result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
#     input_ids,token_type_ids,attention_mask,labels=zip(*result)
#     labels=torch.tensor(labels)
#     max_length=max([len(t) for t in input_ids])
    
#     input_ids_padded=padding(input_ids,max_length)
#     token_type_ids_padded=padding(token_type_ids,max_length)
#     attention_mask_padded=padding(attention_mask,max_length)
#     return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels

    # if padding = 'max_length' is used in tokenizer.encode_plust fuction, then there is no need to pad the sequences to the same size.
    result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
    input_ids,token_type_ids,attention_mask,labels=zip(*result)
    return torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask),torch.tensor(labels)
    


# In[3]:


class BertForSquad(nn.Module):
    def __init__(self,config):
        super(BertForSquad,self).__init__()
        # To determine whether the token is the starting point or end point of the answer, that is actually a two category question.
        self.num_labels=config.num_labels
        self.config=config
        model_config=BertConfig.from_pretrained(config.bert_path)
        self.bert=BertModel.from_pretrained(config.bert_path,config=model_config)
        for param in self.bert.parameters():
            param.requires_grad=True
        self.gru = []
        for i in range(config.gru_layers):
            self.gru.append(
                nn.GRU(
                    config.hidden_size if i == 0 else config.gru_hidden_size * 2,
                    config.gru_hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                )
            )

        self.gru = nn.ModuleList(self.gru)
        self.qa_outputs = nn.Linear(config.gru_hidden_size * 2, config.num_labels)
    def forward(self,x,initial_hidden_state):
        input_ids,token_type_ids,attention_mask=x[0],x[1],x[2]
        outputs=self.bert(input_ids,attention_mask,token_type_ids)
        # get the output of every token
        sequence_output=outputs[0]
        for gru in self.gru:
            try:
                gru.flatten_parameters()
            except:
                pass
            sequence_output, h_n = gru(sequence_output,initial_hidden_state)
        logits=self.qa_outputs(sequence_output)
        # get all start logits and end logits separately.
        start_logits,end_logits=logits.split(1,dim=-1)
        start_logits=start_logits.squeeze(-1)
        end_logits=end_logits.squeeze(-1)
        # end_logits shape (batch_size,sequence_len)
        return start_logits,end_logits


# In[4]:


class Model_Config():
    def __init__(self,):
        self.bert_path="./pretrained_model/bert-base-uncased"
        self.train_data_path='./squad_v2/train-v2.0.json'
        self.dev_data_path='./squad_v2/dev-v2.0.json'
        self.num_labels=2
        self.hidden_size=768
        self.learning_rate=2.0e-5
        self.bert_learning_rate=2e-5
        self.other_learning_rate=5e-5
        self.save_path='./save_model/customized_bert_for_squad.pt'
        self.out_file='./output_evaluate/evaluate_prediction.json'
        self.require_improvement=10000
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eps=1e-8
        self.batch_size=32
        self.num_epochs=1
        self.dropout=0.3
        self.max_sequence_len=256
        self.max_a_len=20
        self.gru_layers=1
        self.gru_hidden_size=384
# class Model_Config():
#     def __init__(self,):
#         self.bert_path='/home/aistudio/work/Squad-main/pretrained_model'
#         self.train_data_path='/home/aistudio/work/Squad-main/data/squad/train-v2.0.json'
#         self.dev_data_path='/home/aistudio/work/Squad-main/data/squad/dev-v2.0.json'
#         self.num_labels=2
#         self.hidden_size=768
#         self.learning_rate=2.0e-5
#         self.bert_learning_rate=2e-5
#         self.other_learning_rate=2e-5
#         self.save_path='/home/aistudio/work/Squad-main/save_model/customized_bert_for_squad.pt'
#         self.out_file='/home/aistudio/work/Squad-main/output_evaluate/evaluate_prediction.json'
#         self.require_improvement=10000
#         self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
#         self.eps=1e-8
#         self.batch_size=32
#         self.num_epochs=1
#         self.dropout=0.3
#         self.max_sequence_len=256
#         self.max_a_len=20
#         self.gru_layers=1
#         self.gru_hidden_size=384


# In[5]:


def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]


# In[6]:


def predict_for_dataset(config,model,dev_dataset,is_test=False):
    model.eval()
    
    total_loss=0.0
    total_batches=0
    
    evaluate_output={}
    
    exact_scores={}
    f1_scores={}
    
    qid_has_answer=set()
    qid_no_answer=set()
    
    with torch.no_grad():
        for index,(output,label) in tqdm(enumerate(dev_dataset)):
            input_ids,token_type_ids,attention_mask=output['input_ids'],output['token_type_ids'],output['attention_mask']
            prepared_input_ids=torch.tensor([input_ids]).to(config.device)
            prepared_token_type_ids=torch.tensor([token_type_ids]).to(config.device)
            prepared_attention_mask=torch.tensor([attention_mask]).to(config.device)
            
            qid=output['qas_id']
            
            labels=torch.tensor([label]).to(config.device)
            
            x=(prepared_input_ids,prepared_token_type_ids,prepared_attention_mask)
            initial_hidden_state=torch.zeros(2,prepared_input_ids.shape[0],config.gru_hidden_size).to(config.device)
            start_logits,end_logits=model(x,initial_hidden_state)
            
            start_loss=torch.nn.functional.cross_entropy(start_logits,labels[:,0])
            end_loss=torch.nn.functional.cross_entropy(end_logits,labels[:,1])
            combined_loss=(start_loss+end_loss)/2
            total_loss+=combined_loss.item()
            total_batches+=1
            
            start_probas,end_probas=torch.softmax(start_logits,dim=-1)[0],torch.softmax(end_logits,dim=-1)[0]
            
            start_end, score = None, -1
            for start, p_start in enumerate(start_probas):
                for end, p_end in enumerate(end_probas):
                    if end >= start and end < start + config.max_a_len:
                        if p_start * p_end > score:
                            start_end = (start, end)
                            score = p_start * p_end
            start, end = start_end
            pred_answer=dev_dataset.tokenizer.convert_tokens_to_string(dev_dataset.tokenizer.convert_ids_to_tokens(input_ids[start:end+1],skip_special_tokens=True))
            
            evaluate_output[qid]=pred_answer
            if not is_test:
                answer=output['answer_text']
                if answer=='' or answer==None:
                    qid_no_answer.add(qid)
                else:
                    qid_has_answer.add(qid)

                if qid in exact_scores:
                    exact_scores[qid]=max(exact_scores[qid],compute_exact(answer, pred_answer))
                else:
                    exact_scores[qid]=compute_exact(answer, pred_answer)
                if qid in f1_scores:
                    f1_scores[qid] = max(f1_scores[qid],compute_f1(answer, pred_answer))
                else:
                    f1_scores[qid] = compute_f1(answer, pred_answer)
    if not is_test:
        total_result=make_eval_dict(exact_scores, f1_scores, qid_list=None)
        has_answer_result=make_eval_dict(exact_scores, f1_scores, qid_list=qid_has_answer)
        no_answer_result=make_eval_dict(exact_scores, f1_scores, qid_list=qid_no_answer)
        merge_eval(total_result,has_answer_result,prefix='HasAns')
        merge_eval(total_result,no_answer_result,prefix='NoAns')
        print('evaluation_performace:',total_result)
    else:
        total_result={}
    fw = open(config.out_file, 'w', encoding='utf-8')
    R = json.dumps(evaluate_output, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()
    return total_loss/total_batches,total_result


# In[7]:


def evaluate_during_training(config,model,dev_iter):
    model.eval()
    total_loss=0.0
    total_batches=0
    with torch.no_grad():
        for index,(input_ids,token_type_ids,attention_mask,labels) in tqdm(enumerate(dev_iter)):
            input_ids=input_ids.to(config.device)
            token_type_ids=token_type_ids.to(config.device)
            attention_mask=attention_mask.to(config.device)
            labels=labels.to(config.device)
            x=(input_ids,token_type_ids,attention_mask)
            initial_hidden_state=torch.zeros(2,input_ids.shape[0],config.gru_hidden_size).to(config.device)
            start_logits,end_logits=model(x,initial_hidden_state)
            start_loss=torch.nn.functional.cross_entropy(start_logits,labels[:,0])
            end_loss=torch.nn.functional.cross_entropy(end_logits,labels[:,1])
            combined_loss=(start_loss+end_loss)/2
            total_loss+=combined_loss.item()
            total_batches+=1
    return total_loss/total_batches


# In[8]:


def train(config,model,train_iter,dev_iter):
    model.train()
    print(config.device)
    bert_param_optimizer=list(model.bert.named_parameters())
    bert_params=list(map(id,model.bert.parameters()))
    other_param_optimizer=[(n,p) for n,p in model.named_parameters() if id(p) not in bert_params]
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters=[
        {'params':[p for n,p in bert_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01,'lr':config.bert_learning_rate},
        {'params':[p for n,p in bert_param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0,'lr':config.bert_learning_rate},
        {'params':[p for n,p in other_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01,'lr':config.other_learning_rate},
        {'params':[p for n,p in other_param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0,'lr':config.other_learning_rate}]
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.learning_rate,eps=config.eps)
    total_batch=0
    best_dev_loss=float('inf')
    last_improve=0
    flag=False # record whether the model is not learning any more
    for epoch in range(config.num_epochs):
        epoch_start_time=time.time()
        print('Epoch:{}/{}'.format(epoch+1,config.num_epochs))
        batched_loss=0
        for index,(input_ids,token_type_ids,attention_mask,labels) in tqdm(enumerate(train_iter)):
            batch_start_time=time.time()
            input_ids=input_ids.to(config.device)
            token_type_ids=token_type_ids.to(config.device)
            attention_mask=attention_mask.to(config.device)
            labels=labels.to(config.device)
            x=(input_ids,token_type_ids,attention_mask)
            initial_hidden_state=torch.zeros(2,input_ids.shape[0],config.gru_hidden_size).to(config.device)
            start_logits,end_logits=model(x,initial_hidden_state)
            model.zero_grad()
            start_loss=torch.nn.functional.cross_entropy(start_logits,labels[:,0])
            end_loss=torch.nn.functional.cross_entropy(end_logits,labels[:,1])
            total_loss=(start_loss+end_loss)/2
            total_loss.backward()
            optimizer.step()
            batched_loss+=total_loss.item()
            if total_batch%3000==0 and total_batch!=0:
                avg_dev_loss=evaluate_during_training(config,model,dev_iter)
                if avg_dev_loss<best_dev_loss:
                    best_dev_loss=avg_dev_loss
                    torch.save(model.state_dict(),config.save_path)
                    last_improve=total_batch
                print('Epoch:{},Batch:{},Avg_train_loss:{},Avg_dev_loss:{}'.format(epoch+1,index+1,batched_loss/(index+1),avg_dev_loss))
                model.train()
            total_batch+=1
            print('Epoch:{},Batch:{},Avg_train_loss:{}'.format(epoch+1,index+1,batched_loss/(index+1)))
            print('Time cost for one batch:',time.time()-batch_start_time)
            if total_batch-last_improve>config.require_improvement:
                print("No optimization for a long time, auto_stopping...")
                flag=True
                break
        print('Time cost for one epoch:',time.time()-epoch_start_time)
        if flag:
            break   


# In[ ]:


model_config=Model_Config()
start_time=time.time()
train_dataset=SquadDataset(data_path=model_config.train_data_path,vocab_path=model_config.bert_path,max_length=model_config.max_sequence_len)
dev_dataset=SquadDataset(data_path=model_config.dev_data_path,vocab_path=model_config.bert_path,max_length=model_config.max_sequence_len)
print('building_time:',time.time()-start_time)

processor=multiprocessing.cpu_count()
train_dataloader=DataLoader(train_dataset,batch_size=model_config.batch_size,collate_fn=collate_func,num_workers=processor)
dev_dataloader=DataLoader(dev_dataset,batch_size=model_config.batch_size,collate_fn=collate_func,num_workers=processor)

model=BertForSquad(model_config)
if os.path.exists(model_config.save_path):
    model.load_state_dict(torch.load(model_config.save_path,map_location=model_config.device))
model.to(model_config.device)
train(model_config,model,train_dataloader,dev_dataloader)


# In[ ]:


model_config=Model_Config()
dev_dataset=SquadDataset(data_path=model_config.dev_data_path,vocab_path=model_config.bert_path,max_length=model_config.max_sequence_len)
model=BertForSquad(model_config)
if os.path.exists(model_config.save_path):
    model.load_state_dict(torch.load(model_config.save_path,map_location=model_config.device))
model.to(model_config.device)
dev_avg_loss,dev_pred_performance=predict_for_dataset(model_config,model,dev_dataset)
print(dev_avg_loss)
print(dev_pred_performance)


# In[9]:


def predict_for_single_example(model,tokenizer,question,context,config,verbose=False):
    model.eval()
    inputs=tokenizer.encode_plus(question, context, max_length=config.max_sequence_len, truncation=True, padding='max_length', return_tensors='pt')
    token_ids=inputs['input_ids'].tolist()[0]
    
    input_ids=inputs['input_ids'].to(config.device)
    token_type_ids=inputs['token_type_ids'].to(config.device)
    attention_mask=inputs['attention_mask'].to(config.device)
    
    x=(input_ids,token_type_ids,attention_mask)
    initial_hidden_state=torch.zeros(2,input_ids.shape[0],config.gru_hidden_size).to(config.device)
    start_logits,end_logits=model(x,initial_hidden_state)
    
    start_probas,end_probas=torch.softmax(start_logits,dim=-1)[0],torch.softmax(end_logits,dim=-1)[0]
            
    start_end, score = None, -1
    for start, p_start in enumerate(start_probas):
        for end, p_end in enumerate(end_probas):
            if end >= start and end < start + config.max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    pred_answer=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_ids[start:end+1],skip_special_tokens=True))
    
    if verbose:
        print(f'Question:{question}')
        print(f'Context:{context}')
        if pred_answer=='':
            print('There is no answer in the context for this question')
        else:
            print(f'Answer:{pred_answer}')
    return pred_answer


# In[10]:


model_config=Model_Config()
tokenizer=BertTokenizer.from_pretrained(model_config.bert_path)
model=BertForSquad(model_config)
if os.path.exists(model_config.save_path):
    model.load_state_dict(torch.load(model_config.save_path,map_location=torch.device('cpu')))
model.to(model_config.device)


# # original context
# To eat is one human need ,and to wear clothes is another human need.People wear clothes for protection and for decoration.There are many kinds of materials for making clothes .Wool is one kind .It comes from sheep,as well as some other animals. Clothes made of wool are very light and warm.In warm countries people like to wear cotton clothes ,because they are soft, cool and comfortable .Cotton clothes are also cheaper than wool garments.Two other materials are silk and linen .Now synthetic materials are also common ,and they are cheaper than natural fabrics, because they are more comfortable than synthetics.
# 
# # original question and answer
# 1、why does person wear clothes? 
# for protection and for decoration. 
# 2、where does wool come from? 
# It comes from sheep,as well as some other animals.
# 3、why In warm countries people like to wear cotton clothes? 
# because they are soft, cool and comfortable . 
# 4、why synthetic materials are cheaper than natural fabrics?
# because they are more comfortable than synthetics 
# 5、which is cheaper between cotton clothes and wool garments? 
# Cotton clothes
# 6、how are clothes made of wool ?
# They are very light and warm

# In[11]:


questiones=["why does person wear clothes?","where does wool come from?","why In warm countries people like to wear cotton clothes?","why synthetic materials are cheaper than natural fabrics?","which is cheaper between cotton clothes and wool garments?","how are clothes made of wool?"]
context="To eat is one human need ,and to wear clothes is another human need.People wear clothes for protection and for decoration.There are many kinds of materials for making clothes .Wool is one kind .It comes from sheep,as well as some other animals. Clothes made of wool are very light and warm.In warm countries people like to wear cotton clothes ,because they are soft, cool and comfortable .Cotton clothes are also cheaper than wool garments.Two other materials are silk and linen .Now synthetic materials are also common ,and they are cheaper than natural fabrics, because they are more comfortable than synthetics."
for index,question in enumerate(tqdm(questiones)):
    pred_answer=predict_for_single_example(model,tokenizer,question,context,model_config,verbose=True)


# # customized_bert_model_for_squad
# We build a customized bert model for squad examples, here you can understand how to build a **model** on base of bert, how to build a **dataset** or **dataloader** for squad similar type of examples，how to build the **train, evaluate and predict** function for squad similar type of examples, some of the code is borrowed from **run_squad.py** scripts, you can build a complete model for question answering type of projects with the template code here. 
# 
# ## Packages
# - Transformers 3.5.0
# - Torch
# 
# ## The process is the following:
# - 1) Build a dataset and dataloader inherited from torch.utils.data.Dataset/Dataloader to process squad data
# 
# - 2) Build a powerful model by adding GRU layers and classification layers on top of bert.
# 
# - 3) Build two predict functions, one is for single example prediction, the other is for dataset prediction or evaluation.
# 
# - 4) Build an evaluate function which can be used during training to evaluate the performance of the model.
# 
# - 5) Build a powerful train function to train the model, the specific evaluation index will be shown out, the model will be saved in predefined condition.
# 
# 
# ## Pretrained model
# You can download the pretrained weights from the [link](https://huggingface.co/bert-large-uncased-whole-word-masking/tree/main)
# You can also download the pretrained weights from the [link](https://huggingface.co/bert-base-uncased)
# 
# ## Special code
# ```python
# # Using multiprocessing function to deal with large number of examples
# class SquadDataset(Dataset):
#     def __init__(self,data_path,vocab_path,max_length=384):
#         super(SquadDataset,self).__init__()
#         self.data_path=data_path
#         self.data_samples=self.get_samples(data_path)
#         self.tokenizer=BertTokenizer.from_pretrained(vocab_path)
#         self.max_length=max_length
#     def get_examples_from_data(self,data):
#         examples=[]
#         title=data['title']
#         for paragraph in data['paragraphs']:
#             context=paragraph['context']
#             for qa in paragraph['qas']:
#                 qid=qa['id']
#                 question=qa['question']
#                 if "is_impossible" in qa:
#                     is_impossible=qa['is_impossible']
#                 else:
#                     is_impossible=len(qa['answers'])==0
#                 if is_impossible:
#                     current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':''}
#                     examples.append(current_example)
#                 else:
#                     current_answer_collectors=[]
#                     for answer in qa['answers']:
#                         text=answer['text']
#                         if text in current_answer_collectors:
#                             continue
#                         else:
#                             current_answer_collectors.append(text)
#                         answer=answer['answer_start']
#                         current_example={'qas_id':qid,'question_text':question,'context_text':context,'is_impossible':is_impossible,'answer_text':text}
#                         examples.append(current_example)
#         return examples
#     def get_samples(self,data_path):
#         with open(data_path,'r',encoding='utf-8') as f:
#             dataset=json.load(f)
#             all_data=dataset['data']
#         processor=multiprocessing.cpu_count()
#         p=Pool(processor)
#         result_list=p.map(self.get_examples_from_data,all_data)
#         all_examples=[]
#         for examples in result_list:
#             all_examples+=examples
#         p.close()
#         return all_examples
# # The collate function is used to pad the input sequences to the same length in dataloader
# def collate_func(batch):
#     def padding(indices,max_length,pad_idx=0):
#         pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
#         return torch.tensor(pad_indices)
#     # if padding = 'max_length' is used in tokenizer.encode_plust fuction, then there is no need to pad the sequences to the same size.
#     result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
#     input_ids,token_type_ids,attention_mask,labels=zip(*result)
#     return torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask),torch.tensor(labels)
# 
#     # if padding = 'max_length' is not used in tokenizer.encode_plust fuction, then you should pad the sequences to the same size.
#     
# #     result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
# #     input_ids,token_type_ids,attention_mask,labels=zip(*result)
# #     labels=torch.tensor(labels)
# #     max_length=max([len(t) for t in input_ids])
#     
# #     input_ids_padded=padding(input_ids,max_length)
# #     token_type_ids_padded=padding(token_type_ids,max_length)
# #     attention_mask_padded=padding(attention_mask,max_length)
# #     return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels
# 
# # The following code is used to evaluate the fitness of the model.
# def normalize_answer(s):
#   """Lower text and remove punctuation, articles and extra whitespace."""
#   def remove_articles(text):
#     regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
#     return re.sub(regex, ' ', text)
#   def white_space_fix(text):
#     return ' '.join(text.split())
#   def remove_punc(text):
#     exclude = set(string.punctuation)
#     return ''.join(ch for ch in text if ch not in exclude)
#   def lower(text):
#     return text.lower()
#   return white_space_fix(remove_articles(remove_punc(lower(s))))
# 
# def get_tokens(s):
#   if not s: return []
#   return normalize_answer(s).split()
# 
# def compute_exact(a_gold, a_pred):
#   return int(normalize_answer(a_gold) == normalize_answer(a_pred))
# 
# def compute_f1(a_gold, a_pred):
#   gold_toks = get_tokens(a_gold)
#   pred_toks = get_tokens(a_pred)
#   common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#   num_same = sum(common.values())
#   if len(gold_toks) == 0 or len(pred_toks) == 0:
#     # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#     return int(gold_toks == pred_toks)
#   if num_same == 0:
#     return 0
#   precision = 1.0 * num_same / len(pred_toks)
#   recall = 1.0 * num_same / len(gold_toks)
#   f1 = (2 * precision * recall) / (precision + recall)
#   return f1
# 
# # The following function is used to do prediction by customized bert model.
# def predict_for_single_example(model,tokenizer,question,context,config,verbose=False):
#     model.eval()
#     inputs=tokenizer.encode_plus(question, context, max_length=config.max_sequence_len, truncation=True, padding='max_length', return_tensors='pt')
#     token_ids=inputs['input_ids'].tolist()[0]
#     
#     input_ids=inputs['input_ids'].to(config.device)
#     token_type_ids=inputs['token_type_ids'].to(config.device)
#     attention_mask=inputs['attention_mask'].to(config.device)
#     
#     x=(input_ids,token_type_ids,attention_mask)
#     initial_hidden_state=torch.zeros(2,input_ids.shape[0],config.gru_hidden_size).to(config.device)
#     start_logits,end_logits=model(x,initial_hidden_state)
#     
#     start_probas,end_probas=torch.softmax(start_logits,dim=-1)[0],torch.softmax(end_logits,dim=-1)[0]
#             
#     start_end, score = None, -1
#     for start, p_start in enumerate(start_probas):
#         for end, p_end in enumerate(end_probas):
#             if end >= start and end < start + config.max_a_len:
#                 if p_start * p_end > score:
#                     start_end = (start, end)
#                     score = p_start * p_end
#     start, end = start_end
#     pred_answer=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_ids[start:end+1],skip_special_tokens=True))
#     
#     if verbose:
#         print(f'Question:{question}')
#         print(f'Context:{context}')
#         if pred_answer=='':
#             print('There is no answer in the context for this question')
#         else:
#             print(f'Answer:{pred_answer}')
#     return pred_answer
# ```
