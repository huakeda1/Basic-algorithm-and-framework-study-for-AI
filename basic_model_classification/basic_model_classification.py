#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os

import tensorflow as tf
import time
import numpy as np

# load_data()函数返回两个元祖（tuple）对象，第一个是训练集，第二个是测试集。
with np.load('./data/mnist.npz') as f:
    x, y = f['x_train'], f['y_train']
    x_val, y_val = f['x_test'], f['y_test']

x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换成整形张量
x_val = 2 * tf.convert_to_tensor(x_val, dtype=tf.float32) / 255. - 1
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)  # 转换成整形张量

print(x.shape, y.shape)
print(x_val.shape, y_val.shape)
BATCH_SIZE = 512
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建数据集对象
train_dataset = train_dataset.shuffle(buffer_size=30000).batch(BATCH_SIZE)  # 批量训练
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))  # 构建数据集对象
val_dataset = val_dataset.batch(BATCH_SIZE)  # 批量训练

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),  # 隐藏层1
    tf.keras.layers.Dense(128, activation='relu'),  # 隐藏层2
    tf.keras.layers.Dense(10)  # 输出层，输出节点数为10
])
save_dir = './data/callbacks/checkpoints'
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


def train_step(inputs, labels, model, optimizer, loss_func, train_acc, train_loss):
    with tf.GradientTape() as tape:
        inputs = tf.reshape(inputs, (-1, 28 * 28))
        predictions = model(inputs)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(labels, predictions)


def test_step(inputs, labels, model, loss_func, test_acc, test_loss):
    inputs = tf.reshape(inputs, (-1, 28 * 28))
    predictions = model(inputs)
    loss = loss_func(labels, predictions)
    test_loss(loss)
    test_acc(labels, predictions)


class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_rate = tf.cast(initial_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_rate * self.decay_rate ** (step / self.decay_steps)


EPOCHS = 100
best_avg_loss = 100
total_steps = EPOCHS * x.shape[0] // BATCH_SIZE
exponential_decay = ExponentialDecay(initial_rate=0.001, decay_steps=int(0.1 * total_steps), decay_rate=0.96)
optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay, clipnorm=2)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')

for epoch in range(EPOCHS):
    step = 0
    start_time = time.time()

    train_acc.reset_states()
    train_loss.reset_states()
    test_acc.reset_states()
    test_loss.reset_states()

    for inputs, labels in train_dataset:
        train_step(inputs, labels, model, optimizer, loss_func, train_acc, train_loss)
        step += 1
        print('epoch:{},batch:{},train_avg_loss:{},train_avg_acc:{}'.format(epoch + 1,
                                                                            step, train_loss.result(),
                                                                            train_acc.result()))
    for inputs, labels in val_dataset:
        test_step(inputs, labels, model, loss_func, test_acc, test_loss)
    print("epoch:{},train_avg_loss:{},train_avg_acc:{},test_avg_loss:{},test_avg_acc:{}".format(epoch + 1,
                                                                                                train_loss.result(),
                                                                                                train_acc.result(),
                                                                                                test_loss.result(),
                                                                                                test_acc.result()))
    if (epoch + 1) % 1 == 0:
        if train_loss.result() < best_avg_loss:
            best_avg_loss = train_loss.result()
            ckpt_save_path = checkpoint_manager.save()
            print('model is saved to {} for {} with best_avg_loss {}'.format(ckpt_save_path, epoch + 1, best_avg_loss))
            print('Time taken for one epoch is {}'.format(time.time() - start_time))
