#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('./home/aistudio/work')


# In[2]:


# the following code is mainly used to unzip the required data to specific folder.
# import os,zipfile
# src_file='home.zip'
# zf=zipfile.ZipFile(src_file)
# zf.extractall('./home')
# zf.close


# In[3]:


import transformers
print(transformers.__version__)


# In[4]:


get_ipython().system(' ls ./home/aistudio/data/data56340')


# In[5]:


get_ipython().system(' ls ./home/aistudio/work')


# In[6]:


import torch
from torch import nn
class Config:
    def __init__(self):
        self.hidden_size=768
        self.num_attention_heads=12
        self.attention_probs_dropout_prob=0.1
class BertPooler(nn.Module):
    def __init__(self,config):
        super(BertPooler,self).__init__()
        self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.activation=nn.Tanh()
    def forward(self,hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states.shape 为[batch_size, seq_len, hidden_dim]
        first_token_tensor=hidden_states[:,0]
        pooled_output=self.dense(first_token_tensor)
        pooled_output=self.activation(pooled_output)
        return pooled_output


# In[7]:


config=Config()
bertpooler=BertPooler(config)
input_tensor=torch.ones([8,50,768])
output_tensor=bertpooler(input_tensor)
assert output_tensor.shape==torch.Size([8,768])


# In[8]:


# internal function from tokenizer
from typing import List,Optional,Tuple
def build_inputs_with_special_tokens(self,token_ids_0:List[int],token_ids_1:Optional[List[int]]=None)->List[int]:
    if token_ids_1 is None:
        return [self.cls_token_id]+token_ids_0+[self.sep_token_id]
    cls=[self.cls_token_id]
    sep=[self.sep_token_id]
    return cls + token_ids_0 + sep + token_ids_1 + sep


# In[9]:


from transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained('./home/aistudio/data/data56340')
inputs_1=tokenizer('欢迎大家来到后厂理工学院学习.')
print(inputs_1)
inputs_2=tokenizer('欢迎大家来到后厂理工学院学习','认识新朋友是一件快乐的事情.')
print(inputs_2)
inputs_3=tokenizer.encode('欢迎大家来到后厂理工学院学习','认识新朋友是一件快乐的事情.')
print(inputs_3)
inputs_4=tokenizer.build_inputs_with_special_tokens(inputs_3)
print(inputs_4)


# In[10]:


# 将每个输入的数据句子中15%的概率随机抽取token，在这15%中的80%概论将token替换成[MASK]，如上图所示，15%中的另外10%替换成其他token，比如把‘理’换成‘后’，15%中的最后10%保持不变，就是还是‘理’这个token。

# 之所以采用三种不同的方式做mask，是因为后面的fine-tuning阶段并不会做mask的操作，为了减少pre-training和fine-tuning阶段输入分布不一致的问题，所以采用了这种策略。
# MLM output layer definition
class BertLMPredictionHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.transform=BertPredictionHeadTransform(config)
        # 在nn.Linear操作过程中的权重和bert输入的embedding权重共享
        # Embedding层和FC层权重共享，Embedding层中和向量 v 最接近的那一行对应的词，会获得更大的预测概率。
        # 实际上，Embedding层和FC层有点像互为逆过程。
        self.decoder=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.bias=nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias=self.bias
    def forward(self,hidden_states):
        hidden_states=self.transform(hidden_states)
        hidden_states=self.decoder(hidden_states)
        return hidden_states
# 只考虑MLM任务，通过BertForMaskedLM完成预训练，loss为CrossEntropyLoss
# 同时考虑MLM和NSP，通过BertForPreTraining完成预训练，loss为CrossEntropyLoss
# as for NSP， self.seq_relationship=nn.Linear(config.hidden_size,2) 


# In[11]:


# DAPT：领域自适应预训练(Domain-Adaptive Pretraining)
# TAPT：任务自适应预训练(Task-Adaptive Pretraining


# In[12]:


# mask token处理
""" Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
def mask_token(inputs:torch.Tensor,tokenizer:BertTokenizer,args)->Tuple[torch.Tensor,torch.Tensor]:
    if tokenizer.mask_token is None:
        raise ValueError('This tokenizer does not have a mask token which is necessary for masked language model. Remove the --mlm flag if you want to use this tokenizer.')
    labels=inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix=torch.full(labels.shape,args.mlm_probability)
    # filter the exist special token which will not be masked anymore.
    special_tokens_mask=[tokenizer.get_special_tokens_mask(val,already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,dtype=torch.bool),value=0.0)
    # filter the exist pad token which will not be masked anymore
    if tokenizer.pad_token is not None:
        padding_mask=labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask,value=0.0)
    # get out the possible masked position with 1.0 which means 15% of all pure tokens will be picked out for relevant masking.
    masked_indices=torch.bernoulli(probability_matrix).bool()
    # we only need the masked position to compute loss while the other token ids are set to be -100
    labels[~masked_indices]=-100
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced=torch.bernoulli(torch.full(labels.shape,0.8)).bool()&masked_indices
    inputs[indices_replaced]=tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% of the time, we replace masked input tokens with random word
    indices_random=torch.bernoulli(torch.full(labels.shape,0.5)).bool()&masked_indices&~indices_replaced
    random_words=torch.randint(len(tokenizer),labels.shape,dtype=torch.long)
    inputs[indices_random]=random_words[indices_random]
    
    # The rest of the time(10%) we keep the masked input tokens unchanged
    return inputs,labels
        


# In[13]:


from transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained('./home/aistudio/data/data56340')
txt = 'AI Studio是基于百度深度学习平台飞桨的人工智能学习与实训社区，提供在线编程环境、免费GPU算力、海量开源算法和开放数据，帮助开发者快速创建和部署模型。'
inputs_all=tokenizer(txt)
pre_inputs=torch.tensor([inputs_all['input_ids']])
print(pre_inputs)
class Args:
    def __init__(self):
        self.mlm_probability = 0.15
args=Args()
inputs,labels=mask_token(pre_inputs,tokenizer,args)
print(inputs)
print(labels)


# # large scale model training strategy
# 
# # gradient accumulation
# # 一般在单卡GPU训练时采用，防止显存溢出
# if args.max_steps>0:
#     t_total=args.max_steps
#     args.num_train_epochs=args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
# else:
#     t_total=len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs
#     
# # for i, (inputs, labels) in enumerate(training_set):
# #   loss = model(inputs, labels)                    # 计算loss
# #   loss = loss / accumulation_steps                # Normalize our loss (if averaged)
# #   loss.backward()                                 # 反向计算梯度，累加到之前梯度上
# #   if (i+1) % accumulation_steps == 0:             
# #       optimizer.step()                            # 更新参数
# #       model.zero_grad()                           # 清空梯度
# 
# # Nvidia 混合精度工具apex
# if args.fp16:
#     try:
#         from apex import amp
#     except ImportError:
#         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training")
#     model,optimizer=amp.initialize(model,optimizer,opt_level=args.fp16_opt_level)
# 
# # multi-gpu training (should be after apex fp16 initialization)
# if args.n_gpu>1:
#     model=torch.nn.DataParallel(model)
# 
# # distributed traing (should be after apex fp16 initialization)
# if args.local_rank != -1:
#     model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
#     
# # 基于Transformer结构的大规模预训练模型预训练和微调都会采用wramup的方式
# # scheduler =get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

# In[14]:


# 在预训练模型训练的开始阶段，BERT模型对数据的初始分布理解很少，在第一轮训练的时候，模型的权重会迅速改变。如果一开始学习率很大，非常有可能对数据产生过拟合的学习，后面需要很多轮的训练才能弥补，会花费更多的训练时间。但模型训练一段时间后，模型对数据分布已经有了一定的学习，这时就可以提升学习率，能够使得模型更快的收敛，训练也更加稳定，这个过程就是warmup，学习率是从低逐渐增高的过程。
# 当BERT模型训练一定时间后，尤其是后续快要收敛的时候，如果还是比较大的学习率，比较难以收敛，调低学习率能够更好的微调。


# In[16]:


# train process
import os
import tqdm
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)

def train(args,train_dataset,model:PreTrainedModel,tokenizer:BertTokenizer)->Tuple[int,float]:
    if args.local_rank in [-1,0]:
        tb_writer=SummaryWriter()
    args.train_batch_size=args.per_gpu_batch_size*max(1,args.n_gpu)
    # 补齐 pad
    def collate(examples:List[torch.tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples,batch_first=True)
        return pad_sequence(examples,batch_first=True,padding_value=tokenizer.pad_token_id)
    train_sampler=RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    # create dataloader for training
    train_dataloader=DataLoader(train_dataset,sampler=train_sampler,batch_size=args.train_batch_size,collate_fn=collate)
    # prepare gradient accumulation
    if args.max_steps>0:
        t_total=args.max_steps
        args.num_train_epochs=args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total=len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs
    # load the model
    model=model.module if hasattr(model,'module') else model # take care of distribute/parallel training
    model.resize_token_embeddings(len(dataloader))
    # Prepare optimizer and schedule(linear warmup and decay)
    no_decay=['bias','LayerNorm.weight']
    optimizer_grouped_parameters=[{'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay':args.weight_decay},{'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0}]
    optimizer=AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)
    # check if saved optimizer or scheduler state exist
    if (args.model_name_or_path 
        and os.path.isfile(os.path.join(args.model_name_or_path,'optimizer.pt'))
       and os.path.isfile(os.path.join(args.model_name_or_path,'scheduler.pt'))):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path,'optimizer.pt')))
        scheduler.load_state_dict(torch.laod(os.path.join(args.model_name_or_path,'scheduler.pt')))
    # 混合精度训练
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
        model,optimizer=amp.initialize(model,optimizer,opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu>1:
        model=torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank!=-1:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    # display log information before training
    logger.info("***** Running training *****")
    logger.info('Num examples =%d',len(train_dataset))
    logger.info("Num Epochs =%d",args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU=%d",args.per_gpu_batch_size)
    logger.info("Total train batch size(w.parallel,distribute&accumulation)=%d",
                args.train_batch_size*args.gradient_accumulation_steps*
                (torch.distributed.get_world_size() if args.local_rank!=-1 else 1),)
    logger.info("Gradient Accumulation steps=%d",args.gradient_accumulation_steps)
    logger.info("Total optimization steps=%d",t_total)
    
    global_step=0
    epochs_trained=0
    steps_trained_in_current_epoch=0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to global step of last saved checkpoint from model path
            checkpoint_suffix=args.model_name_or_path.split('-')[-1].split('/')[0]
            global_step=int(checkpoint_suffix)
            epochs_trained=global_step//(len(train_dataloader)//args.gradient_accumulation_steps)
            steps_trained_in_current_epoch=global_step%(len(train_dataloader)//args.gradient_accumulation_steps)
            logger.info("Continuing training from checkpoint, will skip to saved global step")
            logger.info("Continuing training from epcoh %d",epochs_trained)
            logger.info("Continuing training from global step %d",global_step)
            logger.info("Will skip the first %d step in the first epoch",steps_trained_in_current_epoch)
        except ValueError:
            logger.info(" Starting fine_tuning")
    tr_loss,logging_loss=0.0,0.0
    model.zero_grad()
    train_iterator=trange(epochs_trained,int(args.num_train_epochs),desc='Epoch',disable=args.local_rank not in [-1,0])
    set_seed(args) # Added here for reproducibility
    for epoch in train_iterator:
        epoch_iterator=tqdm(train_dataloader,desc='Iteration',disable=args.local_rank not in [-1,0])
        if args.local_rank!=-1:
            train_sampler.set_epoch(epoch)
        for step,batch in enumerate(epoch_iterator):
            # skip past any already trained step if resuming training
            if steps_trained_in_current_epoch >0:
                steps_trained_in_current_epoch -= 1
                continue
            # 对输入数据进行mask处理
            inputs,labels=mask_tokens(batch,tokenizer,args) if args.mlm else (batch,batch)
            inputs=inputs.to(args.device)
            labels=labels.to(args.device)
            model.train()
            outputs=model(inputs,masked_lm_labels=labels) if args.mlm else model(inputs,labels=labels)
            loss=outputs[0]
            if args.n_gpu>1:
                loss=loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss+=loss.item()
            if (step+1)%args.gradient_accumulation_steps==0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm(amp.master_params(optimizer),args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step+=1
            if agrs.local_rank in [-1,0] and args.logging_steps>0 and global_step%args.logging_steps==0:
                # log metrics
                if args.local_rank==-1 and args.evaluate_during_training:
                    # only evaluate when single GPU otherwise metrics may not average well
                    results=evaluate(args,model,tokenizer)
                    for key,value in results.items():
                        tb_writer.add_scaler("eval_{}".format(key),value,global_step)
                tb_writer.add_scaler('lr',scheduler.get_lr()[0],global_step)
                tb_writer.add_scaler('loss',(tr_loss-logging_loss)/args.logging_steps,global_step)
                logging_loss=tr_loss
            if args.local_rank in [-1,0] and args.save_steps>0 and global_step%args.save_steps==0:
                checkpoint_predix='checkpoint'
                # save model check point
                output_dir=os.path.join(args.outout_dir,"{}-{}".format(checkpoint_prefix,global_step))
                os.makedirs(output_dir,exist_ok=True)
                model_to_save=(model.module if hasattr(model,"module") else model)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args,os.path.join(output_dir,'training_args.bin'))
                logger.info('Saving model checkpoint to %s',output_dir)
                
                _rotate_checkpoints(args,checkpoint_prefix)
                
                torch.save(optimizer.state_dict(),os.path.join(output_dir,'optimizer.pt'))
                torch.save(scheduler.state_dict(),os.path.join(output_dir,'scheduler.pt'))
                logger.info('Saving optimizer and scheduler states to %s',output_dir)
            if args.max_steps>0 and global_step>args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps>0 and global_step>args.max_steps:
            train_iterator.close()
            break
    if args.local_rank in [-1,0]:
        tb_writer.close()
    return global_step,tr_loss/global_step


# ```python
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/scalar_example')
# for i in range(10):
#     writer.add_scalar('quadratic', i**2, global_step=i)
#     writer.add_scalar('exponential', 2**i, global_step=i)
# ```

# In[ ]:


from transformers import get_linear_schedule_with_warmup


# ## Bert pretraining skills study
# We study the skills around the **MLM** and **NSP** and learn how to do MLM and NSP, learn how to get **bertpool** output,learn how to use **BertTokenizer**, replicate the function of **mask token** operation, go through all the process of **training**.
# 
# ### Main Content
# - How to create and use BertPooler 
# - How to use BertTokenizer
# - Reference code for understanding BertForMaskedLM
# - Introduction about DAPT and TAPT
# - How to mask token for MLM
# - Large scale model training strategy
# - Learn the whole training code and process
# 
# ### Packages
# - torch
# - transformers
# - typing
# - apex
# - logging
# - tensorboardX
# - tqdm
# 
# ### Important functions
# - nn.Module
# - nn.Linear()
# - nn.parameter()
# - torch.full()
# - torch.eq()
# - torch.tensor(dtype=torch.bool)
# - torch.masked_fill()
# - torch.bernoulli()
# - torch.randint()
# - torch.nn.DataParallel()
# - torch.nn.parallel.DistributedParallel()
# - torch.utils.data.DataLoader()
# - torch.nn.utils.rnn.pad_sequence()
# - hasattr()
# - AdamW()
# - logging.getLogger().info()
# - logging.basicConfig()
# - SummaryWriter().add_scaler()
# - SummaryWriter().close()
# - os.path.isfile()
# - os.path.join()
# - for step,batch in enumerate(tqdm(dataloader))
# - trange means tqdm(range())
# - epoch_iterator.close()
# - loss.backward()
# - torch.nn.utils.clip_grad_norm_()
# - optimizer.step()
# - scheduler.step()
# - model.zero_grad()
# - os.makedirs(exist_ok=True)
# - model.save_pretrained()
# - tokenizer.save_pretrained()
# - torch.save(optimizer.state_dict(),filedir)
# - get_linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps)
# 
# ### Special code
# ```python
# # class BertLMPredictionHead segment 
# self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
# self.bias = nn.Parameter(torch.zeros(config.vocab_size))
# self.decoder.bias = self.bias
# 
# # mask token segment
# probability_matrix = torch.full(labels.shape, args.mlm_probability)
# special_tokens_mask = [
#     tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
# ]
# probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
# if tokenizer._pad_token is not None:
#     padding_mask = labels.eq(tokenizer.pad_token_id)
#     probability_matrix.masked_fill_(padding_mask, value=0.0)
# masked_indices = torch.bernoulli(probability_matrix).bool()
# labels[~masked_indices] = -100  # We only compute loss on masked tokens
# 
# # train process
# 
# # 补齐pad and create dataloader
# def collate(examples: List[torch.Tensor]):
#     if tokenizer._pad_token is None:
#         return pad_sequence(examples, batch_first=True)
#     return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
# 
# train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
# )
# 
# # Prepare optimizer and schedule (linear warmup and decay)
# no_decay = ["bias", "LayerNorm.weight"]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": args.weight_decay,
#     },
#     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
# ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
# )
# 
# # Load in optimizer and scheduler states
# optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
# scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
# 
# ```

# In[ ]:




