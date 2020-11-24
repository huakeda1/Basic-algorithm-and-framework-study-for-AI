# Named Entity Recognition(Inference)
Named Entity Recognition (NER) is the task of classifying tokens according to a class, for example, identifying a token as a person, an organisation or a location. An example of a named entity recognition dataset is the CoNLL-2003 dataset, which is entirely based on that task. If you would like to fine-tune a model on an NER task, you may leverage the **run_ner.py (PyTorch)**, **run_pl_ner.py (leveraging pytorch-lightning)** or the **run_tf_ner.py (TensorFlow)** scripts.

## Packages
- Transformers 3.5.0
- Torch

## The process is the following:
- 1) Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it with the weights stored in the checkpoint.

- 2) Define the label list with which the model was trained on.

- 3) Define a sequence with known entities, such as “Hugging Face” as an organisation and “New York City” as a location.

- 4) Split words into tokens so that they can be mapped to predictions. We use a small hack by, first, completely encoding and decoding the sequence, so that we’re left with a string that contains the special tokens.

- 5) Encode that sequence into IDs (special tokens are added automatically).

- 6) Retrieve the predictions by passing the input to the model and getting the first output. This results in a distribution over the 9 possible classes for each token. We take the argmax to retrieve the most likely class for each token.

- 7) Zip together each token with its prediction and print it.

## Pretrained model
You can download the pretrained weights from the [link](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)

## Special code
```python
model=AutoModelForTokenClassification.from_pretrained('./pretrained_model',return_dict=True)
tokenizer=AutoTokenizer.from_pretrained('./pretrained_model')
tokens=tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs=tokenizer.encode(sequence,return_tensors='pt')
outputs=model(inputs).logits
# outputs shape (1,32,9)
print(outputs.shape)
predictions=outputs.argmax(dim=2)
# predictions shape (1,32)
print(predictions.shape)
```