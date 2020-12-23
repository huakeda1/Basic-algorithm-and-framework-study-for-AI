# customized_bert_model_for_squad
We build a customized bert model for squad examples, here you can understand how to build a **model** on base of bert, how to build a **dataset** or **dataloader** for squad similar type of examples，how to build the **train, evaluate and predict** function for squad similar type of examples, some of the code is borrowed from **run_squad.py** scripts, you can build a complete model for question answering type of projects with the template code here. 

## Packages
- Transformers 3.5.0
- Torch

## The process is the following:
- 1) Build a dataset and dataloader inherited from torch.utils.data.Dataset/Dataloader to process squad data

- 2) Build a powerful model by adding GRU layers and classification layers on top of bert.

- 3) Build two predict functions, one is for single example prediction, the other is for dataset prediction or evaluation.

- 4) Build an evaluate function which can be used during training to evaluate the performance of the model.

- 5) Build a powerful train function to train the model, the specific evaluation index will be shown out, the model will be saved in predefined condition.


## Pretrained model
You can download the pretrained weights from the [link](https://huggingface.co/bert-large-uncased-whole-word-masking/tree/main)
You can also download the pretrained weights from the [link](https://huggingface.co/bert-base-uncased)

## Special code
```python
# Using multiprocessing function to deal with large number of examples
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
# The collate function is used to pad the input sequences to the same length in dataloader
def collate_func(batch):
    def padding(indices,max_length,pad_idx=0):
        pad_indices=[item+[pad_idx]*max(0,max_length-len(item)) for item in indices]
        return torch.tensor(pad_indices)
    # if padding = 'max_length' is used in tokenizer.encode_plust fuction, then there is no need to pad the sequences to the same size.
    result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
    input_ids,token_type_ids,attention_mask,labels=zip(*result)
    return torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask),torch.tensor(labels)

    # if padding = 'max_length' is not used in tokenizer.encode_plust fuction, then you should pad the sequences to the same size.
    
#     result=[(output['input_ids'],output['token_type_ids'],output['attention_mask'],label) for output,label in batch]
#     input_ids,token_type_ids,attention_mask,labels=zip(*result)
#     labels=torch.tensor(labels)
#     max_length=max([len(t) for t in input_ids])
    
#     input_ids_padded=padding(input_ids,max_length)
#     token_type_ids_padded=padding(token_type_ids,max_length)
#     attention_mask_padded=padding(attention_mask,max_length)
#     return input_ids_padded,token_type_ids_padded,attention_mask_padded,labels

# The following code is used to evaluate the fitness of the model.
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

# The following function is used to do prediction by customized bert model.
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
```