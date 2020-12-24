# Ensemble Technology For  Question Answering(Inference)

We build an template ensemble model for squad dataset, outputs from different models are weighted and summed to get the finial result, this method used here can also be applied to similar scenarios. 

## Packages
- Transformers 3.5.0
- Torch
- numpy

## Pretrained model
You can download the pretrained weights from the [link](https://huggingface.co/mrm8488/bert-medium-finetuned-squadv2)

## Special code
```python
# the following code is mainly used to get average weighted probs from different models 
def get_ensembled_result(many_start_probs,many_end_probs,mode='equal'):
    model_nums=len(many_start_probs)
    if mode=='equal':
        ranks=[1.0/model_nums]*model_nums
    elif mode=='ranks':
        ranks=[index/(model_nums*(model_nums+1)*0.5) for index in range(1,model_nums+1)]
    # get the average combined prob from several models
    avg_start_probs,avg_end_probs=0,0
    reference_combined_prob=0.0
    for i in range(model_nums):
        avg_start_probs+=many_start_probs[i]*ranks[i]
        avg_end_probs+=many_end_probs[i]*ranks[i]
        
        combined_matrix=np.matmul(many_start_probs[i][:,np.newaxis],many_end_probs[i][np.newaxis,:])*ranks[i]**2
        max_combined_prob=np.max(combined_matrix)
        # the following code is used to get best scored result from all model predictions
        if max_combined_prob>reference_combined_prob:
            reference_combined_prob=max_combined_prob
            start,end=np.argwhere(combined_matrix==max_combined_prob)[0]
            model_index,max_answer_start,max_answer_end,max_combined_prob=i,start,end,max_combined_prob       
    # the following code is used to get the average combined prob from several models
    total_combined_avg_result=np.matmul(avg_start_probs[:,np.newaxis],avg_end_probs[np.newaxis,:])
    avg_combined_prob=np.max(total_combined_avg_result)
    answer_start,answer_end=np.argwhere(total_combined_avg_result==avg_combined_prob)[0]
    return model_index,answer_start,answer_end
def get_ensembled_answer(models,tokenizers,question,context,max_sequence_len=384,verbose=False,device='cpu'):
    all_inputs=[tokenizer.encode_plus(question, context, max_length=max_sequence_len, truncation=True, padding='max_length', return_tensors='pt') for tokenizer in tokenizers]
    all_start_probs,all_end_probs=[],[]
    with torch.no_grad():
        for index,model in enumerate(models):
            all_inputs[index].to(device)
            result=model(**all_inputs[index])
            all_start_probs.append(torch.nn.functional.softmax(result.start_logits,dim=1).detach().cpu().numpy()[0])
            all_end_probs.append(torch.nn.functional.softmax(result.end_logits,dim=1).detach().cpu().numpy()[0])
    model_index,answer_start,answer_end=get_ensembled_result(all_start_probs,all_end_probs,mode='equal')
    answer=tokenizers[model_index].decode(all_inputs[model_index]['input_ids'][0,answer_start:answer_end+1],skip_special_tokens=True)
    if verbose:
        print('question:',question)
        print('context:',context)
        print('answer:',answer)
    return answer
```