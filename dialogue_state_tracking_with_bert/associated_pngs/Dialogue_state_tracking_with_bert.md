# Dialogue_state_tracking_with_bert

## dataset

### class Turn

#### methods

##### to_dict

##### from_dict

#### properties

##### turn_id

##### transcript

##### turn_label

##### belief_state

##### system_acts

##### system_transcript

##### asr

##### num

### class Dialogue

#### methods

##### to_dict

##### from_dict

#### properties

##### dialogue_id

##### turns

### class Dataset

#### properties

##### dialogues

#### methods

##### to_dict

##### from_dict

##### iter_turns

##### evaluate_preds

def evaluate_preds(self,preds):
    request=[]
    inform=[]
    joint_goal=[]
    fix={'centre':'center','areas':'area','phone number':'number'}
    i=0
    for d in self.dialogues:
        # inform states of dialogue will be possibly updated as turns going on
        pred_state={}
        for t in d.turns:
            gold_request=set([(s,v) for s,v in t.turn_labels if s=='request'])
            gold_inform=set([(s,v) for s,v in t.turn_labels if s!='request'])
            pred_request=set([(s,v) for s,v in preds[i] if s=='request'])
            pred_inform=set([(s,v) for s,v in preds[i] if s!='request'])
            request.append(gold_request==pred_request)
            inform.append(gold_inform==pred_inform)
            gold_recovered=set()
            pred_recovered=set()
            for s,v in pred_inform:
                pred_state[s]=v
            for b in t.belief_state:
                for s,v in b['slots']:
                    if b['act']!='request':
                        gold_recovered.add((b['act'],fix.get(s.strip(),s.strip()),fix.get(v.strip(),v.strip())))
            for s,v in pred_state.items():
                pred_recovered.add(('inform',s,v))
            joint_goal.append(gold_recovered==pred_recovered)
            i+=1
    return {'turn_inform':np.mean(inform),'turn_request':np.mean(request),'joint_goal':np.mean(joint_goal)}

###### turn_inform

###### turn_request

###### joint_goal

### class Ontology

#### properties

##### dialogue_id

##### turns

#### methods

##### to_dict

##### from_dict

### class format

class Ontology:
  def __init__(self,slots=None,values=None,num=None):
    self.slots=slots or {}
    self.values=values or {}
    self.num=num or {}
  def to_dict(self):
    return {'slots':self.slots,'values':self.values,'num':self.num}
  @classmethod
  def from_dict(cls,d):
    return cls(**d)

## main

### train

if opts.do_train:
  model=Model.from_scratch(opts.bert_model)
  model.move_to_device(opts)
  model.run_train(dataset,ontology,opts)
  del model
  torch.cuda.empty_cache()


### eval

if opts.do_eval:
  model=Model.from_model_path(os.path.abspath(opts.output_dir))
  model.move_to_device(opts)
  model.run_dev(dataset,ontology,opts)
  model.run_test(dataset,ontology,opts)

### args

from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data_dir',type=Path,required=True)
parser.add_argument('--bert_model',type=str,required=True,choices=['bert-base-uncased','bert-large-uncased'])
parser.add_argument('--output_dir',type=Path,required=True)
parser.add_argument('--epochs',type=int,default=25)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_proportion', type=float, default=0.1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--random_oversampling',action='store_true')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
parser.add_argument('--do_eval', action='store_true', help='Whether to run evaluation.')
opts=parser.parse_args()

### load_dataset

dataset={}
dataset['train']=Dataset.from_dict(read_json(base_path/'train.json'))
ontology=Ontology.from_dict(read_json(base_path/'ontology.json'))

### read_json

with open(fp) as json_file:
  data=json.load(json_fil)

### device

opts.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opts.n_gpus=torch.cuda.device_count() if str(opts.device)=='cuda' else 0


## models

### turn_to_examples

#### ontology

read from json file

##### slots

##### values

#### context

' '.join([turn.system_transcript]+[SEP]+[' '.join(turn.transcript)])

#### candidate

slot+' = '+value

#### input_text

' '.join([CLS,context,SEP,candidate,SEP])

#### label

int((slot,value) in set([(s,v) for s,v in turn.turn_label]))

#### token_type_ids

alist=tokenized_text[:-1]
sent1_len=len(alist)-alist[-1::-1].index(SEP)-1
sent2_len=len(tokenized_text)-sent1_len
token_type_ids=[0]×sent1_len+[1]×sent2_len

#### input_ids

tokenized_text=tokenizer.tokenize(input_text)
input_ids=tokenizer.convert_tokens_to_ids(tokenized_text)

#### output

[(slot1,value1,input_ids1,token_type_ids1,label1),(slot2,value2,input_ids2,token_type_ids2,label2),...]

### class Model

#### properties

##### tokenizer

##### bert

#### methods

##### from_scratch

###### tokenizer

tokenizer=BertTokenizer.from_pretrained(bert_model)

###### bert

bert=BertForSequenceClassification.from_pretrained(bert_mode)

##### from_model_path

###### tokenizer

tokenizer=BertTokenizer.from_pretrained(output_model_path)

###### bert

bert=BertForSequenceClassification.from_pretrained(output_model_path)

##### move_to_device

bert.to(args.device)
if args.n_gpus>1:
  bert=torch.nn.DataParallel(bert)

##### init_optimizer

param_optimizer=list(self.bert.named_parameters())
no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
optimizer_group_parameters=[{'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},{'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}]
optimizer=AdamW(optimizer_group_parameters,lr=args.learning_rate,eps=args.adam_epsilon)

scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_proportion×num_train_iters,num_training_steps=num_train_iters)

optimizer.zero_grad()


##### run_train

###### train_examples

turns=list(dataset['train'].iter_turns())
train_examples=[turn_to_examples(t,ontology,tokenizer) for t in turns]
train_examples=list(itertools.chain.from_iterable(train_examples))

###### random_sampling

negative_examples=0
positive_examples=0
for example in train_examples:
	if example[-1]==0:
    	negative_examples.append(example)
        if example[-1]==1:
    	positive_examples.append(example)
nb_negatives, nb_positives = len(negative_examples), len(positive_examples)
sampled_positive_examples=random.choices(positive_examples,k=int(nb_negatives/8))
train_examples=sampled_positive_examples+negative_examples

###### num_train_iters

num_train_iters=args.epochs×len(train_examples)/args.batch_size/args.gradient_accumulation_steps

###### train_avg_loss

class RunningAverage():
  def __init__(self):
    self.steps=0
    self.total=0
  def update(self,val):
    self.steps+=1
    self.total+=val
  def __call__(self):
    return self.total/float(self.steps)
train_avg_loss=RunningAverage()
train_avg_loss.update(loss.item())
current_loss=train_avg_loss()

###### train_process

iterations=0
for epoch in range(args.epochs):
  random.shuffle(train_examples)
  iterations+=1    pbar=tqdm(range(0,len(train_examples)),args.batch_size))
  for i in pbar:
    batch=train_examples[i:i+args.batch_size]
_,_,input_ids,token_type_ids,labels=list(zip(*batch))
input_ids,_=pad(input_ids,args.device)
token_type_ids,_=pad(token_type_ids,args.device)
labels=torch.LongTensor(labels).to(args.device)
loss,logits=model(input_ids,token_type_ids=token_type_ids,labels=labels)
if args.n_gpus>1:
  loss=loss.mean()
if args.gradient_accumulation_steps>1:
  loss=loss/args.gradient_accumulation_steps
 loss.backward()
 train_avg_loss.update(loss.item())
 pbar.update(1)
 pbar.set_postfix_str(f'Train loss:{train_avg_loss()}')
 if iterations%args.gradient_accumulation_steps==0:
   self.optimizer.step()
   self.scheduler.step()
   self.optimizer.zero_grad()

###### pad

def pad(seqs,device,pad=0):
  lens=[len(s) for s in seqs]
  max_len=max(lens)
  padded=torch.LongTensor([s+[pad]×(max_len-l) for s,l in zip(seqs,lens)])
  return padded.to(device),lens

##### predict_turn

def predict_turn(self, turn, ontology, args, threshold=0.5):
    model, tokenizer = self.bert, self.tokenizer
    batch_size = args.batch_size
    was_training = model.training
    model.eval()
    preds = []
    examples = turn_to_examples(turn, ontology, tokenizer)
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        slots, values, input_ids, token_type_ids, _ = list(zip(*batch))
        # Padding and Convert to Torch Tensors
        input_ids, _ = pad(input_ids, args.device)
        token_type_ids, _ = pad(token_type_ids, args.device)
        # Forward Pass
        logits = model(input_ids, token_type_ids=token_type_ids)[0]
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().data.numpy()
        # Update preds
        for j in range(len(batch)):
            if probs[j] >= threshold:
                preds.append((slots[j], values[j]))
    if was_training:
        model.train()
    return preds

##### run_dev

def run_dev(dataset,ontology,args):
  turns=list(dataset['dev'].iter_turns())
  preds=[self.predict_turn(t,ontology,args) for t in turns]
  return dataset['dev'].evaluate_preds(preds)

##### run_test

def run_dev(dataset,ontology,args):
  turns=list(dataset['test'].iter_turns())
  preds=[self.predict_turn(t,ontology,args) for t in turns]
  return dataset['test'].evaluate_preds(preds)

##### save

def save(self,output_model_path,verbose=True):
  model,tokenizer=self.bert,self.tokenizer
  model_to_save=model.module if hasattr(model,'module') else model
    output_model_file=output_model_path/WEIGHTS_NAME
  output_config_file=output_model_path/CONFIG_NAME
  torch.save(model_to_save.state_dict(),output_model_file)
  model_to_save.config.to_json_file(output_config_file)
  tokenizer.save_vocabulary(output_model_path)
  if verbose:
    print('Saved the model, the config and the tokenizer')

##### get_n_grams

pp=0
for p in list(model.parameters()):
  nn=1
  for j in list(p.size()):
    nn=nn*j
  pp+=nn
