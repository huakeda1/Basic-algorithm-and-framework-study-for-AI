# Bert_NLU Based on CrossWOZ

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `[["Inform","Hotel","Price", "cheap"]]`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. An MLP takes bert word embeddings as input and classify the tag label. If you set `context=true` in config file, pooled output of bert word embeddings of utterances of last three turn will be unsqueezed,'repeated',concatenated and provide context information with embeddings of current utterance for slot tagging.
- For each of the other dialog acts, such as `["Request","Hotel","Address",""]`, another MLP takes embeddings of `[CLS]` of current utterance as input and do the binary classification. If you set `context=true` in config file, pooled output of bert word embeddings of utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for intent classification.  

We fine-tune BERT parameters on crosswoz.

## Data
We use the crosswoz data (`crosswoz/[train|val|test].json.zip`).

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data
On `nlu_bert_crosswoz/crosswoz` dir:  python preprocess.py [mode]
output processed data on `data/[mode]_data/` dir.

#### Train a model
On `nlu_bert_crosswoz/src` dir:python train.py --config_path crosswoz/configs/[config_file]
The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. 

#### Test a model
On `nlu_bert_crosswoz/src` dir:python test.py --config_path crosswoz/configs/[config_file]
The result (`output.json`) will be saved under `output_dir` of config_file.
The test is summarized as below:  
8476 samples test   
As for intents(loss:0.0025028494149128117;precision:94.63;Recall: 96.75;F1: 95.68)  
As for slots(loss:0.07231740884810911;Precision:97.20;Recall:93.71;F1: 95.43)  
As for overall(Precision:96.19;Recall:94.87;F1:95.52)  

#### analyze test data
On `nlu_bert_crosswoz/src` dir:python analyse.py  
You can have a complete evaluation of the trained model.  
all(P:0.9618917797664909,R:0.9486789805938742,F1:0.9552396927694888)  
不独立多领域(samples:3490,[P:95.98, R:93.93, F1:94.94])  
单领域(samples:240,[P:97.21, R:97.21, F1:97.21])  
独立多领域(samples:2208,[P:96.31, R:95.82, F1:96.06])  
不独立多领域+交通(samples:1232,[P:95.48, R:94.99, F1:95.24])  
独立多领域+交通(samples:1306,[P:97.09, R:95.33, F1:96.2])  
Intents=['General','Inform','NoOffer','Recommend','Request','Select']  
{'General': [99.39, 99.69, 99.54],'Inform': [96.32, 92.95, 94.6],'NoOffer': [90.24, 96.52, 93.28],'Recommend': [97.94, 98.57, 98.25],'Request': [95.59, 97.79, 96.68],'Select': [80.83, 86.22, 83.44]}   
Domains=['General','出租','地铁','景点','酒店','餐馆']  
{'General': [99.39, 99.69, 99.54],'出租': [99.31, 97.07, 98.17],'地铁': [97.51, 95.33, 96.4],'景点': [96.64, 94.19, 95.4],'酒店': [93.9, 92.76, 93.33],'餐馆': [96.25, 95.48, 95.87]}  

#### make prediction by model
On `nlu_bert_crosswoz/src` dir:python nlu.py  
You can get intents and slots from the input of utterance and relevant context list.

#### Trained model  
We have trained two models: one use context information (last 3 utterances)(`configs/crosswoz_all_context.json`) and the other doesn't (`configs/crosswoz_all.json`) on **all** utterances of crosswoz dataset (`crosswoz/[train|val|test].json.zip`). 
Performance:F1 for model without context is 91.85,F1 for model with context is 95.53.

Models can be download from:

Without context: https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all.zip

With context: https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip

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
self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
self.slot_loss_fct = torch.nn.CrossEntropyLoss()
# here we get masked loss for slot tagging.
if tag_seq_tensor is not None:
    active_tag_loss = tag_mask_tensor.view(-1) == 1
    active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
    active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
    slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)
outputs = outputs + (slot_loss,)
if intent_tensor is not None:
    intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
    outputs = outputs + (intent_loss,)
```
## Core code for data preprocessing
```python
def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

assert mode == 'all' or mode == 'usr' or mode == 'sys'
cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)
data_key = ['train', 'val', 'test']
data = {}
for key in data_key:
    data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
    print('load {}, size {}'.format(key, len(data[key])))

processed_data = {}
all_intent = []
all_tag = []

context_size = 3

tokenizer = BertTokenizer.from_pretrained("./hfl/chinese-bert-wwm-ext")

for key in data_key:
    processed_data[key] = []
    for no, sess in data[key].items():
        context = []
        for i, turn in enumerate(sess['messages']):
            if mode == 'usr' and turn['role'] == 'sys':
                context.append(turn['content'])
                continue
            elif mode == 'sys' and turn['role'] == 'usr':
                context.append(turn['content'])
                continue
            utterance = turn['content']
            # Notice: ## prefix, space remove
            tokens = tokenizer.tokenize(utterance)
            golden = []

            span_info = []
            intents = []
            for intent, domain, slot, value in turn['dialog_act']:
                if intent in ['Inform', 'Recommend'] and '酒店设施' not in slot:
                    if value in utterance:
                        idx = utterance.index(value)
                        idx = len(tokenizer.tokenize(utterance[:idx]))
                        span_info.append(
                            ('+'.join([intent, domain, slot]), idx, idx + len(tokenizer.tokenize(value)), value))
                        token_v = ''.join(tokens[idx:idx + len(tokenizer.tokenize(value))])
                        # if token_v != value:
                        #     print(slot, token_v, value)
                        token_v = token_v.replace('##', '')
                        golden.append([intent, domain, slot, token_v])
                    else:
                        golden.append([intent, domain, slot, value])
                else:
                    intents.append('+'.join([intent, domain, slot, value]))
                    golden.append([intent, domain, slot, value])

            tags = []
            for j, _ in enumerate(tokens):
                tag = ''
                for span in span_info:
                    if j == span[1]:
                        tag = "B+" + span[0]
                        tags.append(tag)
                        break
                    if span[1] < j < span[2]:
                        tag = "I+" + span[0]
                        tags.append(tag)
                        break
                if tag == '':
                    tags.append("O")

            processed_data[key].append([tokens, tags, intents, golden, context[-context_size:]])

            all_intent += intents
            all_tag += tags

            context.append(turn['content'])

    all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
    all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
    print('loaded {}, size {}'.format(key, len(processed_data[key])))
    json.dump(processed_data[key],
              open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

print('sentence label num:', len(all_intent))
print('tag num:', len(all_tag))
print(all_intent)
json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)
json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)
```
## Core code for data postprocessing
```python
def is_slot_da(da):
    tag_da = {'Inform', 'Recommend'}
    not_tag_slot = '酒店设施'
    if da[0] in tag_da and not_tag_slot not in da[2]:
        return True
    return False
def tag2das(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent, domain, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            das.append([intent, domain, slot, value])
        i += 1
    return das

def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split('\+', intent)
        triples.append([intent, domain, slot, value])
    return triples

def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    max_seq_len = tag_logits.size(0)
    das = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
            das.append([intent, domain, slot, value])
    tags = []
    for j in range(1 , max_seq_len -1):
        if tag_mask_tensor[j] == 1:
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    tag_intent = tag2das(ori_word_seq, tags)
    das += tag_intent
    return das
```
## Core code for dataloader
The input data is a formated as:[tokens,tags,intents,golden,context[-context_size:]]  
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
self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
```
```python
class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, pretrained_weights):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]]

            d[4] = self.tokenizer.encode('[CLS] ' + ' [SEP] '.join(d[4]))
            max_context_len = max(max_context_len, len(d[4]))
            context_len.append(len(d[4]))

            if use_bert_tokenizer:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
            else:
                word_seq = d[0]
                tag_seq = d[1]
                new2ori = None
            d.append(new2ori)
            d.append(word_seq)
            d.append(self.seq_tag2id(tag_seq))
            d.append(self.seq_intent2id(d[2]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data['train'])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        accum = ''
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            if (accum + token).lower() == word_seq[j].lower():
                accum = ''
            else:
                accum += token
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(basic_tokens[i]):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                new_tag_seq.append(tag_seq[j])
            if accum == '':
                j += 1
        return split_tokens, new_tag_seq, new2ori

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
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            words = ['[CLS]'] + words + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, 1:sen_len-1] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, 1:sen_len-1] = torch.LongTensor([1] * (sen_len-2))
            for j in intents:
                intent_tensor[i, j] = 1.
            context_len = len(batch_data[i][-5])
            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][-5]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
```
## Core code for evaluation
```python
def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1

if step % check_step == 0:
    train_slot_loss = train_slot_loss / check_step
    train_intent_loss = train_intent_loss / check_step
    print('[%d|%d] step' % (step, max_step))
    print('\t slot loss:', train_slot_loss)
    print('\t intent loss:', train_intent_loss)

    predict_golden = {'intent': [], 'slot': [], 'overall': []}

    val_slot_loss, val_intent_loss = 0, 0
    model.eval()
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None

        with torch.no_grad():
            slot_logits, intent_logits, slot_loss, intent_loss = model.forward(word_seq_tensor,
                                                                               word_mask_tensor,
                                                                               tag_seq_tensor,
                                                                               tag_mask_tensor,
                                                                               intent_tensor,
                                                                               context_seq_tensor,
                                                                               context_mask_tensor)
        val_slot_loss += slot_loss.item() * real_batch_size
        val_intent_loss += intent_loss.item() * real_batch_size
        for j in range(real_batch_size):
            predicts = recover_intent(dataloader, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
                                      ori_batch[j][0], ori_batch[j][-4])
            labels = ori_batch[j][3]

            predict_golden['overall'].append({
                'predict': predicts,
                'golden': labels
            })
            predict_golden['slot'].append({
                'predict': [x for x in predicts if is_slot_da(x)],
                'golden': [x for x in labels if is_slot_da(x)]
            })
            predict_golden['intent'].append({
                'predict': [x for x in predicts if not is_slot_da(x)],
                'golden': [x for x in labels if not is_slot_da(x)]
            })

    for j in range(10):
        writer.add_text('val_sample_{}'.format(j),
                        json.dumps(predict_golden['overall'][j], indent=2, ensure_ascii=False),
                        global_step=step)

    total = len(dataloader.data['val'])
    val_slot_loss /= total
    val_intent_loss /= total
    print('%d samples val' % total)
    print('\t slot loss:', val_slot_loss)
    print('\t intent loss:', val_intent_loss)

    writer.add_scalar('intent_loss/train', train_intent_loss, global_step=step)
    writer.add_scalar('intent_loss/val', val_intent_loss, global_step=step)

    writer.add_scalar('slot_loss/train', train_slot_loss, global_step=step)
    writer.add_scalar('slot_loss/val', val_slot_loss, global_step=step)

    for x in ['intent', 'slot', 'overall']:
        precision, recall, F1 = calculateF1(predict_golden[x])
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

        writer.add_scalar('val_{}/precision'.format(x), precision, global_step=step)
        writer.add_scalar('val_{}/recall'.format(x), recall, global_step=step)
        writer.add_scalar('val_{}/F1'.format(x), F1, global_step=step)

    if F1 > best_val_f1:
        best_val_f1 = F1
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        print('best val F1 %.4f' % best_val_f1)
        print('save on', output_dir)

    train_slot_loss, train_intent_loss = 0, 0
```
## Core code for training
```python
model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
model.to(DEVICE)

if config['model']['finetune']:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': config['model']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['model']['learning_rate'],
                      eps=config['model']['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['model']['warmup_steps'],
                                                num_training_steps=config['model']['max_step'])
else:
    for n, p in model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config['model']['learning_rate'])
for step in range(1, max_step + 1):
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)
        batched_data = tuple(t.to(DEVICE) for t in batched_data)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = batched_data
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None
        _, _, slot_loss, intent_loss = model.forward(word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor,
                                                     intent_tensor, context_seq_tensor, context_mask_tensor)
        train_slot_loss += slot_loss.item()
        train_intent_loss += intent_loss.item()
        loss = slot_loss + intent_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if config['model']['finetune']:
            scheduler.step()  # Update learning rate schedule
        model.zero_grad()
torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
model_path = os.path.join(output_dir, 'pytorch_model.bin')
zip_path = config['zipped_model_path']
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(model_path)
```
## Core code for prediction
```python
class BERTNLU(NLU):
    def __init__(self, mode='all', config_file='./crosswoz/configs/crosswoz_all_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        # config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config_file = config_file
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        data_dir = config['data_dir']
        output_dir = config['output_dir']

        # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # data_dir = os.path.join(root_dir, config['data_dir'])
        # output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'), encoding='utf-8'))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json'), encoding='utf-8'))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                            pretrained_weights=os.path.join("./crosswoz", config['model']['pretrained_weights']))

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(output_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][-4])
        return intent
```