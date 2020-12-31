## sentiment_analysis_bilstm_glove_based
We build an template model of sentiment analysis with word embeddings loaded from pretrained 'glove.6B.50d.txt', you can understand how the original text are transformed into standard sequences used for the inputs of the model, you can almost learn the commonly used processes of dealing with sentiment analysis, you can also build your own model by taking it as a reference.

### Main Pacakages
- tensorflow
- numpy
- os
- tensorflow.keras.preprocessing.text.Tokenizer
- tensorflow.keras.preprocessing.sequence.pad_sequences

### Main Process
- read data from files into label lists and text lists
- tokenize the texts and pad the sequences
- random shufle and split the datasets
- read embedding vectors from glove trained file
- prepare the index embedding matrix for customized words embedding
- build the bidirectional lstm classification model
- train and save the model

### Important functions
- os.path.abspath(__file__)
- os.path.dirname
- os.path.join
- os.listdir
- Tokenizer(num_words=max_words)
- tokenizer.fit_on_texts(texts)
- tokenizer.texts_to_sequences(texts)
- pad_sequences(sequences, maxlen=max_len, padding='post')
- np.random.shuffle(indices),data = data[indices]
- tf.keras.models.Sequential()
- tf.keras.layers.Embedding
- tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
- tf.keras.layers.GlobalAveragePooling1D()
- tf.keras.layers.Dense
- tf.keras.optimizers.Adam
- tf.keras.losses.BinaryCrossentropy(from_logits=False)
- model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
- model.load_weights(model_save_path)
- model.save_weights(model_save_path)
- model.fit(x_train, y_train, epochs=60, batch_size=32, validation_data=(x_val, y_val))

### Pretrained global vectors for word representation
You can get pretrained global vectors for word representation from this [address](https://nlp.stanford.edu/projects/glove/)

### Dataset for aclImdb
You can get the dataset of aclImdb from this [address](http://ai.stanford.edu/~amaas/data/sentiment/)

### Special code
```python
# read data from files into label lists and text lists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imdb_dir = os.path.join(BASE_DIR, 'datasets/aclImdb')
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
print('data from files have already been read into labels and texts.')
# tokenize the text and pad the sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_len, padding='post')
labels = np.asarray(labels)
# random shufle and split the datasets
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_examples]
y_train = labels[:training_examples]
x_val = data[training_examples:training_examples + validation_examples]
y_val = labels[training_examples:training_examples + validation_examples]
# prepare the index embedding matrix
embedding_dim = 50
index_embedding = np.zeros((max_words, embedding_dim))
unk_words = []
for word, i in word_index.items():
    if i < max_words:
        vector = word_embedding.get(word)
        if vector is not None:
            index_embedding[i] = vector
        else:
            unk_words.append(word)
print('{} words are not found in glove embedding matrix.'.format(len(unk_words)))
print(unk_words)
# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_len))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.layers[0].set_weights([index_embedding])
model.layers[0].trainable = False
# train and save the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()
model_save_path = os.path.join(BASE_DIR, 'data/glove_based_tf_model.h5')
if os.path.exists(model_save_path):
    model.load_weights(model_save_path)
    print('model restored from exist model weights')
else:
    print('model is going to be trained from scratch')
history = model.fit(x_train, y_train, epochs=60, batch_size=32, validation_data=(x_val, y_val))
model.save_weights(model_save_path)
```
### Unsolved Error
Failed to call ThenRnnForward with model config: [rnn_mode, rnn_input_mode, rnn_direction_mode]: 2, 0, 0 , [num_layers, input_size, num_units, dir_count, max_seq_length, batch_size, cell_num_units]: [1, 50, 64, 1, 100, 32, 64][[{{node CudnnRNN}}]]  
The problem may be that the version of Cudnn does not match that of tensorlow.