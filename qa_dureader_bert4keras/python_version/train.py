#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
# from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Layer, Dense, Permute, Bidirectional, GRU
from keras.models import Model
from collections import OrderedDict
from utils.config import Config
import sys
import io
from tqdm import tqdm
from utils.evaluate import evaluate as src_evaluate
from utils.data_utils import sequence_padding, load_data, DataGenerator, search_answer_start


class MaskedSoftmax(Layer):
    """
    在序列长度那一维进行softmax，并mask掉padding部分
    """
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)


def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


class data_generator(DataGenerator):
    def __init__(self,tokenizer,max_seq_len,**kwargs):
        super(data_generator,self).__init__(**kwargs)
        self.tokenizer=tokenizer
        self.maxlen=max_seq_len
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            context, question, answers = item[1:]
            token_ids, segment_ids = self.tokenizer.encode(
                question, context, maxlen=self.maxlen
            )
            a = np.random.choice(answers)
            a_token_ids = self.tokenizer.encode(a)[0][1:-1]
            start_index = search_answer_start(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def extract_answer(question, context, model, tokenizer, max_seq_len, max_a_len=16):
    """
    抽取答案函数
    """
    max_q_len = 64
    q_token_ids = tokenizer.encode(question, maxlen=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, maxlen=max_seq_len - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    probas = model.predict([[token_ids], [segment_ids]])[0]
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]


def predict_to_file(infile, out_file, model, tokenizer, max_seq_len):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1], model, tokenizer, max_seq_len)
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()


def evaluate(filename, model, tokenizer, max_seq_len):
    """
    评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json', model, tokenizer, max_seq_len)
    ref_ans = json.load(io.open(filename))
    pred_ans = json.load(io.open(filename + '.pred.json'))
    F1, EM, TOTAL, SKIP = src_evaluate(ref_ans, pred_ans)
    output_result = OrderedDict()
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    return output_result


class Evaluator(keras.callbacks.Callback):
    """
    评估和保存模型
    """
    def __init__(self,model, tokenizer, output_dir, max_seq_len):
        self.best_val_f1 = 0.
        self.model=model
        self.tokenizer=tokenizer
        self.output_dir=output_dir
        self.max_seq_len=max_seq_len

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(os.path.join(data_dir,'dev.json'),self.model, self.tokenzier, self.max_seq_len)
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            self.model.save_weights(os.path.join(self.output_dir,'roberta_best_model.weights'))
            self.model.save(os.path.join(self.output_dir,'roberta_best_model.h5'))
        metrics['BEST_F1'] = self.best_val_f1
        print(metrics)


if __name__ == "__main__":
    args=Config()
    train_data = load_data(os.path.join(args.data_dir,'train.json'))
    tokenizer = Tokenizer(args.dict_path, do_lower_case=True)
    train_generator = data_generator(tokenizer=tokenizer, max_seq_len=args.maxlen, data=train_data, batch_size=args.batch_size, buffer_size=None)

    model = build_transformer_model(args.config_path,args.checkpoint_path)
    output = Bidirectional(GRU(384,return_sequences=True))(model.output)
    output = Dense(2)(output)
    output = MaskedSoftmax()(output)
    output = Permute((2, 1))(output)

    model = Model(model.input, output)
    model.summary()

    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(args.learing_rate),metrics=[sparse_accuracy])

    evaluator = Evaluator(model,tokenizer,args.output_dir,args.maxlen)

    model.fit_generator(train_generator.forfit(),steps_per_epoch=len(train_generator),epochs=args.epochs,verbose=1,callbacks=[evaluator])