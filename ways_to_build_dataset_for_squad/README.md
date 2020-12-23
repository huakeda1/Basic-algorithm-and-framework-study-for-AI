# Different ways of building dataset for squad data.
We study different ways of building dataset for squad data, these ways can also be used in other relevant application, we also make full use of the multiprocessor to process the data, this way of doing can really increase the efficiency of preprocessing.


## Packages
- Torch
- multiprocessing
- math
- json
- transformers
- numpy

## The following ways of building dataset for squad data are studied in this repository:
- 1) Using built-in function from transformers to create the dataloader for squad

- 2) Building a new dataset inherited from torch.utils.data.Dataset.

- 3) Building a new dataset by python yield way.

- 4) Another way of multiprocessing data.


## Special code
```python
# Using built-in function from transformers to create the dataset for squad
tokenizer = BertTokenizer.from_pretrained(bert_dir)
processor = SquadV2Processor()
train_examples = processor.get_train_examples(squad_v2_data_dir,'train-v2.0.json')
# the following code can be used to query the internal element of one class
dir(train_examples[0])
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
# Using multiprocessing to process the squad data
from multiprocessing import Pool
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
# The core code of building a new dataset by python yield way.
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
            
# Another way of multiprocessing data.
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
```
