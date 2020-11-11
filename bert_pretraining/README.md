## Bert pretraining skills study
We study the skills around the **MLM** and **NSP** and learn how to do MLM and NSP, learn how to get **bertpool** output,learn how to use **BertTokenizer**, replicate the function of **mask token** operation, go through all the process of **training**.

### Main Content
- How to create and use BertPooler 
- How to use BertTokenizer
- Reference code for understanding BertForMaskedLM
- Introduction about DAPT and TAPT
- How to mask token for MLM
- Large scale model training strategy
- Learn the whole training code and process

### Packages
- torch
- transformers
- typing
- apex
- logging
- tensorboardX
- tqdm

### Important functions
- nn.Module
- nn.Linear()
- nn.parameter()
- torch.full()
- torch.eq()
- torch.tensor(dtype=torch.bool)
- torch.masked_fill()
- torch.bernoulli()
- torch.randint()
- torch.nn.DataParallel()
- torch.nn.parallel.DistributedParallel()
- torch.utils.data.DataLoader()
- torch.nn.utils.rnn.pad_sequence()
- hasattr()
- AdamW()
- logging.getLogger().info()
- logging.basicConfig()
- SummaryWriter().add_scaler()
- SummaryWriter().close()
- os.path.isfile()
- os.path.join()
- for step,batch in enumerate(tqdm(dataloader))
- trange means tqdm(range())
- epoch_iterator.close()
- loss.backward()
- torch.nn.utils.clip_grad_norm_()
- optimizer.step()
- scheduler.step()
- model.zero_grad()
- os.makedirs(exist_ok=True)
- model.save_pretrained()
- tokenizer.save_pretrained()
- torch.save(optimizer.state_dict(),filedir)
- get_linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps)

### Special code
```python
# class BertLMPredictionHead segment 
self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
self.bias = nn.Parameter(torch.zeros(config.vocab_size))
self.decoder.bias = self.bias

# mask token segment
probability_matrix = torch.full(labels.shape, args.mlm_probability)
special_tokens_mask = [
    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
]
probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
if tokenizer._pad_token is not None:
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
masked_indices = torch.bernoulli(probability_matrix).bool()
labels[~masked_indices] = -100  # We only compute loss on masked tokens

# train process

# 补齐pad and create dataloader
def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
)

# Load in optimizer and scheduler states
optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

```