#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# https://blog.csdn.net/qq_44783177/article/details/108350477
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

results = np.loadtxt('./data/cal_housing.data', delimiter=',', dtype=np.float32)
inputs = results[:, :-1]
labels = results[:, -1][:, np.newaxis]
print(inputs.shape)
print(labels.shape)
column_names = []
with open('./data/cal_housing.domain', mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        if not line.strip():
            continue
        column_names.append(line.split(':')[0])
input_column_name = column_names[:-1]
label_column_name = column_names[-1]
print("input_column_name:", input_column_name)
print("label_column_name:", label_column_name)
train_inputs_all, test_inputs, train_labels_all, test_labels = train_test_split(inputs, labels, test_size=0.2,
                                                                                random_state=0)
train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(train_inputs_all, train_labels_all,
                                                                          test_size=0.1, random_state=0)
print(train_inputs.shape, train_labels.shape)
print(valid_inputs.shape, valid_labels.shape)
print(test_inputs.shape, test_labels.shape)

inputs_scaler = StandardScaler()
train_inputs_scaled = inputs_scaler.fit_transform(train_inputs)
valid_inputs_scaled = inputs_scaler.transform(valid_inputs)
test_inputs_scaled = inputs_scaler.transform(test_inputs)

labels_scaler = StandardScaler()
train_labels_scaled = labels_scaler.fit_transform(train_labels)
valid_labels_scaled = labels_scaler.transform(valid_labels)
test_labels_scaled = labels_scaler.transform(test_labels)

print(test_labels_scaled[:20])
print(test_labels[:20])
print(labels_scaler.mean_)
print(labels_scaler.scale_)

BATCH_SIZE = 8
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs_scaled, train_labels_scaled))  # 构建数据集对象
train_dataset = train_dataset.shuffle(buffer_size=train_inputs_scaled.shape[0]).batch(BATCH_SIZE)  # 批量训练
val_dataset = tf.data.Dataset.from_tensor_slices((valid_inputs_scaled, valid_labels_scaled))  # 构建数据集对象
val_dataset = val_dataset.batch(BATCH_SIZE)  # 批量训练
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs_scaled, test_labels_scaled))  # 构建数据集对象
test_dataset = test_dataset.batch(BATCH_SIZE)  # 批量训练


class CustomizedDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activation
        self.activation_func = tf.keras.layers.Activation(self.activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='uniform', trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        return self.activation_func(x @ self.kernel + self.bias)

    def get_config(self):
        config = {'units': self.units, 'activation': self.activation}
        base_config = super(CustomizedDenseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


model = tf.keras.models.Sequential([
    CustomizedDenseLayer(64, activation='relu', input_shape=train_inputs_scaled.shape[1:]),
    CustomizedDenseLayer(64, activation='relu'),
    CustomizedDenseLayer(1),
])


def train_step(inputs, labels, model, optimizer, loss_func, train_loss, l2_loss_coeff=0.0001):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss_pure = loss_func(labels, predictions)
        loss_regularization = []
        # l2 regularization
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        loss =loss_pure + l2_loss_coeff*loss_regularization
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_pure)


def test_step(inputs, labels, model, loss_func, test_loss):
    predictions = model(inputs)
    loss = loss_func(labels, predictions)
    test_loss(loss)


class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_rate = tf.cast(initial_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_rate * self.decay_rate ** (step / self.decay_steps)

    def get_config(self):
        config = {'initial_rate': self.initial_rate, 'decay_steps': self.decay_steps, 'decay_rate': self.decay_rate}
        base_config = super(ExponentialDecay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


EPOCHS = 200
best_avg_loss = 200
total_steps = EPOCHS * train_inputs_scaled.shape[0] // BATCH_SIZE
exponential_decay = ExponentialDecay(initial_rate=0.0001, decay_steps=int(0.05 * total_steps), decay_rate=0.90)
optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay, clipnorm=2)
loss_func = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


total_steps = EPOCHS * train_inputs_scaled.shape[0] // BATCH_SIZE
loss = tf.keras.losses.MeanSquaredError()

save_dir = './data/callbacks/checkpoints/regression'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_save_dir = save_dir
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_save_dir, max_to_keep=5)
checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    print('Restored from exist model')
else:
    print('Trained from scratch.')

for epoch in range(EPOCHS):
    step = 0
    start_time = time.time()
    train_loss.reset_states()
    test_loss.reset_states()

    for inputs, labels in train_dataset:
        train_step(inputs, labels, model, optimizer, loss_func, train_loss)
        step += 1
        print('epoch:{},batch:{},train_avg_loss:{}'.format(epoch + 1, step, train_loss.result()))
    for inputs, labels in val_dataset:
        test_step(inputs, labels, model, loss_func, test_loss)
    print("epoch:{},train_avg_loss:{},test_avg_loss:{}".format(epoch + 1, train_loss.result(), test_loss.result()))
    if (epoch + 1) % 1 == 0:
        if train_loss.result() < best_avg_loss:
            best_avg_loss = train_loss.result()
            ckpt_save_path = checkpoint_manager.save()
            print('model is saved to {} for {} with best_avg_loss {}'.format(ckpt_save_path, epoch + 1, best_avg_loss))
            print('Time taken for one epoch is {}'.format(time.time() - start_time))

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# metrics = [tf.keras.metrics.MeanAbsoluteError()]
# model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
# model_save_dir = './data/callbacks/checkpoints/regression'
# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir)
# callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_dir, save_best_only=True)]
# history = model.fit(train_inputs_scaled, train_labels_scaled,
#                     steps_per_epoch=int(train_inputs_scaled.shape[0] // BATCH_SIZE), epochs=EPOCHS,
#                     callbacks=callbacks,
#                     validation_data=(valid_inputs_scaled, valid_labels_scaled),
#                     validation_steps=int(valid_labels_scaled.shape[0] // BATCH_SIZE))
#
#
# def plot_learning_curve(history):
#     pd.DataFrame(history.history).plot(figsize=(8, 5))
#     plt.grid(True)
#     plt.gca().set_ylim(0, 1)
#     plt.show()
# plot_learning_curve(history)

