## Transformer basic study
We build a bert and cnn combined fine-tuned model on the base of the transformed pretrained bert model from Google, we know how to transform the pretrained model from the format of tf to pt, we learn how to load pretrained model in new defined network for downstream tasks, we understand the name and weights of layers so as to setup different optimizer values for different parameters of layers, the model we build here can used to do text classification.

### Main Content
- build a text classification model by incorporating pretrained model 
- study name and weights of parameters
- study the way of transforming pretrained weights from the format of tf to pt

### Packages
- torch
- transformers
- zipfile

### Important functions
- nn.Module
- nn.Linear()
- nn.parameter()
- nn.ModuleList()
- nn.Conv1d()
- nn.init.xavier_normal_
- nn.init.constant_
- torch.permute()
- torch.nn.functional.max_pool1d()
- torch.cat()
- nn.Dropout()

### Pretrained models
pretrained wobert model:[link](https://github.com/ZhuiyiTechnology/WoBERT)  
pretrained bert model:[link](https://github.com/ymcui/Chinese-BERT-wwm)

### Special code
```python
# layer parameters initialization 
def init_params(self):
        for m in self.convs:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data,0.1)

# forward calculation
def forward(self,x):
        """
        Args:
            input_ids: token_id
            token_type_ids: 0 means first sentence,1 means second sentence
            attention_mask: 1 means token, 0 means padding
        """
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        
        encoded_layer,_=self.bert(context,token_type_ids=token_type_ids,attention_mask=mask)
        # why the tensor should be permuted?
        encoded_layer=encoded_layer.permute(0,2,1)
        # conved[0] shape (batch_size,n_filters,-1)
        conved=self.convs(encoded_layer)
        # conved[0] shape (batch_size,n_filters)
        max_pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        # cat shape (batch_size,n_filters*len(filter_sizes))
        cat = self.dropout(torch.cat(max_pooled,dim=1))
        # logits shape (batch_size,num_labels)
        logits = self.classifier(cat)
        return logits

# split none bert parameters from the model
bert_params=[]
other_params=[]
for n,p in model.named_parameters():
    if 'bert' in n:
        bert_params.append((n,p))
    else:
        other_params.append((n,p))
bert_names,bert_parameters=zip(*bert_params)
other_names,other_parameters=zip(*other_params)
print(bert_names)
print(other_names)

# unzip the required data to specific folder.
import os,zipfile
src_file='chinese_wobert_L-12_H-768_A-12.zip'
zf=zipfile.ZipFile(src_file)
zf.extractall('./home/aistudio/data/wobert')
zf.close

# transform pretrained weights from the format of tf to pt
%run convert_bert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt \
  --bert_config_file ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_path ./home/aistudio/data/wobert/chinese_wobert_L-12_H-768_A-12/pytorch_model.bin
```