#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

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
print('total training samples:{}'.format(len(labels)))

max_len = 100
training_examples = 20000
validation_examples = 5000
max_words = 20000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_len, padding='post')
labels = np.asarray(labels)
print('data shape:', data.shape)
print('labels shape:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_examples]
y_train = labels[:training_examples]
x_val = data[training_examples:training_examples + validation_examples]
y_val = labels[training_examples:training_examples + validation_examples]

glove_dir = os.path.join(BASE_DIR, 'data/glove.6B.50d.txt')
word_embedding = {}
with open(glove_dir, encoding='utf-8') as f:
    for line in f.readlines():
        if not line.strip():
            continue
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype=np.float32)
        word_embedding[word] = vector
print('found %d word vectors.' % len(word_embedding))
# build our word embedding
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
# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_len))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.layers[0].set_weights([index_embedding])
model.layers[0].trainable = False
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

