# NLU Joint With Bert On CrossWOZ

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `[["Inform","Hotel","Price", "cheap"]]`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. An MLP takes bert word embeddings as input and classify the tag label. If you set `context=true` in config file, pooled output of bert word embeddings of utterances of last three turn will be unsqueezed,'repeated',concatenated and provide context information with bert word embeddings of current utterance for slot tagging.
- For each of the other dialog acts, such as `["Request","Hotel","Address",""]`, another MLP takes embeddings of `[CLS]` of current utterance as input and do the binary classification. If you set `context=true` in config file, pooled output of bert word embeddings of utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for intent classification.  

We fine-tune BERT parameters on crosswoz.

## Data
We use the crosswoz data (`crosswoz_data/[train|val|test].json.zip`).

## Usage

Both user and system utterances are used here to train.

#### Preprocess data
preprocess_data function is used to preprocess the original data and output the following processed data to 'crosswoz_data/'  
1) formated_train_nlu_data.json,formated_test_nlu_data.json,formated_val_nlu_data.json  
2) intent_vocab.json,tag_vocab.json.

#### Load data
Dataloader class is used to load,process and provide batched data for training,evaluating and testing, the original data and processed tensor are both packaged into the batched data so as to make it easy to do evaluating later.

#### Build model
For sequence taging task:    
Pooled output of bert word embeddings of utterances of last three turn will be unsqueezed,'repeated',concatenated and provide context information with bert word embeddings of current utterance for slot tagging.  
For intent classification task:  
Pooled output of bert word embeddings of utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` of current utterance for intent classification.  

#### Train model
The train function is used to train the model, evaluation will be done during the training process,the weight of model will be saved under `save_weight_path` of config class, the model itself will be saved under 'save_model_path' of config class. 

#### Evaluate model
The evaluate function is used to evaluate the model, the loss and score of final result is listed as below:  
As for intents(loss:0.00213;precision:95.63;Recall: 97.28;F1: 96.45)    
As for slots(loss:0.06841;Precision:96.47;Recall:91.463;F1: 93.90)    
As for overall(Precision:96.14;Recall:93.69;F1:94.896)    

#### Make prediction by model  
The predict_intent_slot function is used to get intents and slots from the input of utterance and relevant context list.

Pretrained bert model can be download from the following link:    
[https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main)

## References

## Core code for model
```python
self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
self.slot_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)

nn.init.xavier_uniform_(self.intent_hidden.weight)
nn.init.xavier_uniform_(self.slot_hidden.weight)
nn.init.xavier_uniform_(self.intent_classifier.weight)
nn.init.xavier_uniform_(self.slot_classifier.weight)

outputs = self.bert(input_ids=word_seq_tensor,attention_mask=word_mask_tensor)
sequence_output = outputs[0]
pooled_output = outputs[1]
# context information from pool output of bert model will be processed and incorporated with sequence and pool out of current utterance。  
context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
sequence_output = torch.cat([context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),sequence_output],dim=-1)
pooled_output = torch.cat([context_output, pooled_output], dim=-1)

sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

sequence_output = self.dropout(sequence_output)
slot_logits = self.slot_classifier(sequence_output)
outputs = (slot_logits,)

pooled_output = self.dropout(pooled_output)
intent_logits = self.intent_classifier(pooled_output)
outputs = outputs + (intent_logits,)
```

## Core code for loss
```python
if config.if_intent_weight:
    intent_loss_fct =torch.nn.BCEWithLogitsLoss(pos_weight=dataloader.intent_weight.to(config.device))
else:
    intent_loss_fct = torch.nn.BCEWithLogitsLoss()
# here we get masked loss for slot tagging.
if config.mask_loss:
    active_tag_loss = tag_mask_tensor.view(-1) == 1
        # I made some change for the view function
    active_tag_logits = slot_logits.view(-1, slot_logits.size()[-1])[active_tag_loss]
    active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
else:
    active_tag_logits = slot_logits
    active_tag_labels = tag_seq_tensor
slot_loss = slot_loss_fct(active_tag_logits, active_tag_labels)
intent_loss = intent_loss_fct(intent_logits, intent_tensor)
```
## Core code for data preprocessing
```python
 data_key=['train','val','test']
    intent_vocab=[]
    tag_vocab=[]
    for key in data_key:
        file_name=os.path.join(file_dir,key+'.json.zip')
        zpf=zipfile.ZipFile(file_name,'r')
        data=json.load(zpf.open(key+'.json'))
        sessions=[]
        for num,session in data.items():
            for i,message in enumerate(session["messages"]):
                utterance=message["content"]
                word_seq=tokenizer.tokenize(utterance)
                if message["role"]=="sys" and not include_sys:
                    pass
                else:
                    processed_data=[]
                    slots={}
                    intents=[]
                    golden=[]
                    for intent,domain,slot,value in message["dialog_act"]:
                        if intent in ['Inform','Recommend'] and '酒店设施' not in slot:
                            if value in utterance:
                                idx=utterance.index(value)
                                idx=len(tokenizer.tokenize(utterance[:idx]))
                                new_value=''.join(word_seq[idx:idx+len(tokenizer.tokenize(value))])
                                new_value=new_value.replace('##','')
                                golden.append([intent,domain,slot,new_value])
                                
                                slot_name="+".join([intent,domain,slot])
                                if slot_name not in slots:
                                    slots[slot_name]=[value]
                                else:
                                    slots[slot_name].append(value)
                            else:
                                golden.append([intent,domain,slot,value])
                        else:
                            intent_name='+'.join([intent,domain,slot,value])
                            intents.append(intent_name)
                            intent_vocab.append(intent_name)
                            golden.append([intent,domain,slot,value])                        
                    tag_seq=generate_tags(tokenizer,word_seq,slots)
                    tag_vocab+=tag_seq
                    processed_data.append(word_seq)
                    processed_data.append(tag_seq)
                    processed_data.append(intents)
                    processed_data.append(golden)
                    # attention please copy.deepcopy should be used to prevent data change later effect
                    current_context=[item["content"] for item in session["messages"][0:i] ]
#                     if len(current_context)==0:current_context=['']
                    processed_data.append(current_context)
                    sessions.append(processed_data)
        with open(os.path.join(file_dir,f'formated_{key}_nlu_data.json'),"w",encoding='utf-8') as g:
            json.dump(sessions,g,indent=2,ensure_ascii=False)
        print(os.path.join(file_dir,f'formated_{key}_nlu_data.json'))
    with open(os.path.join(file_dir,'intent_vocab.json'),"w",encoding='utf-8') as h:
        output_intent_vocab=[x[0] for x in dict(Counter(intent_vocab)).items()]
        json.dump(output_intent_vocab,h,indent=2,ensure_ascii=False)
    print(os.path.join(file_dir,'intent_vocab.json'))
    with open(os.path.join(file_dir,'tag_vocab.json'),"w",encoding='utf-8') as j:
        output_tag_vocab=[x[0] for x in dict(Counter(tag_vocab)).items()]
        json.dump(output_tag_vocab,j,indent=2,ensure_ascii=False)
    print(os.path.join(file_dir,'tag_vocab.json'))
```
## Core code for data postprocessing
```python
def is_slot_da(da):
    if da[0] in ['Inform','Recommend'] and '酒店设施' not in da[2]:
        return True
    return False
def get_score(predict_golden):
    TP,FP,FN=0,0,0
    for item in predict_golden:
        predicts=item['predict']
        labels=item['golden']
        for item in predicts:
            if item in labels:
                TP+=1
            else:
                FP+=1
        for item in labels:
            if item not in predicts:
                FN+=1
    precision=1.0*TP/(TP+FP) if TP+FP else 0.0
    recall=1.0*TP/(TP+FN) if TP+FN else 0.0
    F1=2.0*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision,recall,F1
def tag2das(word_seq,tag_seq):
    assert len(word_seq)==len(tag_seq)
    das=[]
    i=0
    while i<len(tag_seq):
        tag=tag_seq[i]
        if tag.startswith('B'):
            intent,domain,slot=tag[2:].split('+')
            value=word_seq[i]
            j=i+1
            while j<len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:]==tag[2:]:
                    if word_seq[j].startswith('##'):
                        value+=word_seq[j][2:]
                    else:
                        value+=word_seq[j]
                    i+=1
                    j+=1
                else:
                    break
            das.append([intent,domain,slot,value])
        i+=1
    return das
def recover_intent(dataloader,intent_logits,tag_logits,tag_mask_tensor,ori_word_seq,new2ori):
    max_seq_len=tag_logits.size(0)
    das=[]
    for j in range(dataloader.intent_dim):
        if intent_logits[j]>0:
            intent,domain,slot,value=re.split('\+',dataloader.id2intent[j])
            das.append([intent,domain,slot,value])
    tags=[]
    for j in range(1,max_seq_len-1):
        if tag_mask_tensor[j]==1:
            value,tag_id=torch.max(tag_logits[j],dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    tag_intent=tag2das(ori_word_seq,tags)
    das+=tag_intent
    return das
```
## Core code for dataloader
The input data is a formated as:[tokens,tags,intents,golden,context] 
The output data before batched is formated as:[tokens,tags,intents,golden,context[token id],new2ori,new_word_seq,tag2id_seq,intent2id_seq]  
The batched data is formated as:[word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor]  
```python
# the following code is applied to consider the deviation of the distribution of labels by giving different weights to diffent label while calculating loss.
self.intent_weight = [1] * len(self.intent2id)
if data_key=='train':
    for intent_id in d[-1]:
        self.intent_weight[intent_id] += 1
if data_key == 'train':
    train_size = len(self.data['train'])
    for intent, intent_id in self.intent2id.items():
        neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
        self.intent_weight[intent_id] = np.log10(neg_pos)
    self.intent_weight = torch.tensor(self.intent_weight)
self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.nn.BCEWithLogitsLoss(pos_weight=dataloader.intent_weight.to(config.device)))
```
```python
class Dataloader:
    def __init__(self, intent_vocab_path, tag_vocab_path, pretrained_weights, max_history=3):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert_policy, e.g. 'bert_policy-base-uncased'
        """
        with open(intent_vocab_path,'r',encoding='utf-8') as f:
            self.intent_vocab=json.load(f)
        with open(tag_vocab_path,'r',encoding='utf-8') as g:
            self.tag_vocab=json.load(g)
        self.intent_dim = len(self.intent_vocab)
        self.tag_dim = len(self.tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(self.intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(self.intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(self.tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(self.tag_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)
        self.max_history=max_history
        self.max_sen_len=0
        self.max_context_len=0

    def load_data(self, data_path, data_key, cut_sen_len=0):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/tests
        :param data:
        :return:
        """
        # data是[tokens, tags, intents, raw_dialog_act, context[-context_size:]]]五个纬度的嵌套列表
        # tokens是jieba切分的得到的词，tags是可以看作词对应的slot标签，
        with open(data_path,'r',encoding='utf-8') as f:
            self.data[data_key]=json.load(f)
        max_context_len=0
        max_sen_len=0
        for d in self.data[data_key]:
            # d = (tokens, tags, intents, raw_dialog_act, context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [" ".join(s.split()[:cut_sen_len]) for s in d[4][-self.max_history:]]

            d[4] = self.tokenizer.encode("[CLS] " + " [SEP] ".join(d[4]))
            
            max_context_len = max(max_context_len, len(d[4]))
            word_seq = d[0]
            tag_seq = d[1]
            new2ori = None
            d.append(new2ori)
            d.append(word_seq)
            d.append(self.seq_tag2id(tag_seq))
            d.append(self.seq_intent2id(d[2]))
            # here sep and cls will be added later
            max_sen_len = max(max_sen_len, len(word_seq)+2)
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key == "train":
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == "train":
            train_size = len(self.data["train"])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (
                    train_size - self.intent_weight[intent_id]
                ) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
            self.max_context_len=max_context_len
            self.max_sen_len=max_sen_len
            print("max sen bert_policy len from train data", self.max_sen_len)
            print("max context bert_policy len from train data", self.max_context_len)

    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags if x in self.tag2id]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_sen_len = max([len(x[-3]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_sen_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_sen_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_sen_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_sen_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        max_context_len = max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros(
            (batch_size, max_context_len), dtype=torch.long)
        context_seq_tensor = torch.zeros(
            (batch_size, max_context_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]  #
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            words = ["[CLS]"] + words + ["[SEP]"]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, 1 : sen_len - 1] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, 1 : sen_len - 1] = torch.LongTensor([1] * (sen_len - 2))
            for j in intents:
                intent_tensor[i, j] = 1.0
            context_len = len(batch_data[i][-5])
            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][-5]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor,word_mask_tensor,tag_seq_tensor,tag_mask_tensor,intent_tensor,context_seq_tensor,context_mask_tensor
    
    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data["train"], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size : (i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
```
## Core code for evaluation
```python
def evaluate(config,model,dataloader,data_key,slot_loss_fct,intent_loss_fct):
    model.eval()
    val_slot_loss,val_intent_loss=0,0
    predict_golden={'intent':[],'slot':[],'overall':[]}
    score_result={'intent':[],'slot':[],'overall':[]}
    for index,(model_inputs,batch_data,num_data) in tqdm(enumerate(dataloader.yield_batches(config.batch_size,data_key))):
        model_inputs=tuple(item.to(config.device) for item in model_inputs)
        word_seq_tensor,word_mask_tensor,tag_seq_tensor,tag_mask_tensor,intent_tensor,context_seq_tensor,context_mask_tensor=model_inputs
        with torch.no_grad():
            slot_logits,intent_logits=model.forward(word_seq_tensor,word_mask_tensor,context_seq_tensor,context_mask_tensor)

            slot_loss,intent_loss=get_total_loss_func(dataloader,config,intent_logits,intent_tensor,slot_logits,tag_seq_tensor,tag_mask_tensor,intent_loss_fct,slot_loss_fct)

        val_slot_loss+=slot_loss.item()*num_data
        val_intent_loss+=intent_loss.item()*num_data
        
        for i in range(num_data):
            predicts=recover_intent(dataloader,intent_logits[i],slot_logits[i],tag_mask_tensor[i],batch_data[i][0],batch_data[i][-4])
            labels=batch_data[i][3]
            predict_golden['overall'].append({'predict':predicts,'golden':labels})
            predict_golden['intent'].append({'predict':[x for x in predicts if not is_slot_da(x)],'golden':[x for x in labels if not is_slot_da(x)]})
            predict_golden['slot'].append({'predict':[x for x in predicts if is_slot_da(x)],'golden':[x for x in labels if is_slot_da(x)]})
    for x in ['intent','slot','overall']:
        precision,recall,F1=get_score(predict_golden[x])
        score_result[x]=[precision,recall,F1]
        print('-'*20+x+'-'*20)
        print('Precision:{},Recall:{},F1:{}'.format(precision,recall,F1))
    avg_slot_loss=val_slot_loss/len(dataloader.data[data_key])
    avg_intent_loss=val_intent_loss/len(dataloader.data[data_key])
    print('val_slot_loss:{}，val_intent_loss:{}'.format(avg_slot_loss,avg_intent_loss))
    return avg_slot_loss,avg_intent_loss,score_result
```
## Core code for training
```python
def train(config,model,dataloader,slot_loss_fct,intent_loss_fct):
    print(config.device)
    bert_param_optimizer=list(model.bert.named_parameters())
    bert_params=list(map(id,model.bert.parameters()))
    other_param_optimizer=[(n,p) for n,p in model.named_parameters() if id(p) not in bert_params]
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters=[
        {'params':[p for n,p in bert_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':config.weight_decay,'lr':config.bert_learning_rate},
        {'params':[p for n,p in bert_param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0,'lr':config.bert_learning_rate},
        {'params':[p for n,p in other_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':config.weight_decay,'lr':config.other_learning_rate},
        {'params':[p for n,p in other_param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0,'lr':config.other_learning_rate}]
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.learning_rate,eps=config.eps)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=config.warmup_steps,num_training_steps=config.max_step)
    
    train_slot_loss,train_intent_loss=0,0
    best_dev_loss=float('inf')
    total_train_samples=0
    for step in tqdm(range(1,config.max_step+1)):
        model.train()
        batched_data=dataloader.get_train_batch(config.batch_size)
        batched_data=tuple(item.to(config.device) for item in batched_data)
        word_seq_tensor,word_mask_tensor,tag_seq_tensor,tag_mask_tensor,intent_tensor,context_seq_tensor,context_mask_tensor=batched_data

        slot_logits,intent_logits=model.forward(word_seq_tensor,word_mask_tensor,context_seq_tensor,context_mask_tensor)

        slot_loss,intent_loss=get_total_loss_func(dataloader,config,intent_logits,intent_tensor,slot_logits,tag_seq_tensor,tag_mask_tensor,intent_loss_fct,slot_loss_fct)
        optimizer.zero_grad()
        total_loss=slot_loss+intent_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        train_slot_loss+=slot_loss.item()*word_seq_tensor.size(0)
        train_intent_loss+=intent_loss.item()*word_seq_tensor.size(0)
        total_train_samples+=word_seq_tensor.size(0)
        scheduler.step()
        if step%config.check_step==0:
            train_slot_loss=train_slot_loss/total_train_samples
            train_intent_loss=train_intent_loss/total_train_samples
            print('current_step{}/total_steps{},train_slot_loss:{},train_intent_loss:{}'.format(step,config.max_step,train_slot_loss,train_intent_loss))
            avg_slot_loss,avg_intent_loss,score_result=evaluate(config,model,dataloader,'val',slot_loss_fct,intent_loss_fct)
            avg_dev_loss=avg_slot_loss+avg_intent_loss
            if avg_dev_loss<best_dev_loss:
                best_dev_loss=avg_dev_loss
                torch.save(model.state_dict(),config.save_weight_path)
                print('model is saved to:{}'.format(config.save_weight_path))
```
## Core code for prediction
```python
def predict_intent_slot(utterance:str,context:list,config,dataloader,model):
    # utterance: str, context: list
    model.eval()

    context_seq = dataloader.tokenizer.encode("[CLS] " + " [SEP] ".join(context[-config.max_history:]))
    
    
    ori_word_seq=dataloader.tokenizer.tokenize(utterance)
    ori_tag_seq = ["O"]*len(ori_word_seq)
    
    intents = []
    da = []
    word_seq,tag_seq,new2ori=ori_word_seq,ori_tag_seq,None

    batch_data=[[ori_word_seq,ori_tag_seq,intents,da,context_seq,new2ori,word_seq,dataloader.seq_tag2id(tag_seq),dataloader.seq_intent2id(intents)]]
    pad_batch=dataloader.pad_batch(batch_data)
    pad_batch=tuple(t.to(config.device) for t in pad_batch)
    
    word_seq_tensor,word_mask_tensor,tag_seq_tensor,tag_mask_tensor,intent_tensor,context_seq_tensor,context_mask_tensor=pad_batch
    with torch.no_grad():
        slot_logits,intent_logits = model.forward(word_seq_tensor,word_mask_tensor,context_seq_tensor,context_mask_tensor)
    das=recover_intent(dataloader,intent_logits[0],slot_logits[0],tag_mask_tensor[0],batch_data[0][0],batch_data[0][-4])
    return das
```