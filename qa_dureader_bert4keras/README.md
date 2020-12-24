# 2020 language and intelligent technology competion:machine reading comprehension task
This evaluation will provide a high-quality Chinese reading comprehension dataset Dureader Robust for real application scenarios. It aims to provide researchers and developers with a platform for academic and technical exchanges, further improve the research level of the machine reading comprehension, and promote the development of technology and application in language understanding and artificial intelligence.
You can get more detailed information from this [link](https://aistudio.baidu.com/aistudio/competition/detail/28)
## Data Introduction
This competition data set contains about 21k questions, including 15K training set, about 1.4k domain development set and 5K test set. The test set includes domain test set and robustness test set. The robustness test set includes hypersensitivity test set, over stability test set and generalization ability test set. All data sets will be divided into 4 parts for users to download.

## Data Sample
The data provided by the platform is in JSON file format, and the example is  shown as follows:

    {
        "data": [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "question": "非洲气候带", 
                                "id": "bd664cb57a602ae784ae24364a602674", 
                                "answers": [
                                    {
                                        "text": "热带气候", 
                                        "answer_start": 45
                                    }
                                ]
                            }
                        ], 
                        "context": "1、全年气温高，有热带大陆之称。主要原因在与赤道穿过大陆中部，位于南北纬30度之间，主要是热带气候，没有温带和寒带… 
                    }, 
                    {
                        "qas": [
                            {
                                "question": "韩国全称", 
                                "id": "a7eec8cf0c55077e667e0d85b45a6b34", 
                                "answers": [
                                    {
                                        "text": "大韩民国", 
                                        "answer_start": 5
                                    }
                                ]
                            }
                        ], 
                        "context": "韩国全称“大韩民国”，位于朝鲜半岛南部，隔“三八线”与朝鲜民主主义人民共和国相邻，面积9.93万平方公理… "
                    }
                ], 
                "title": ""
            }
        ]
    }

## Main Process
- 1) Transform the json format data into list of examples, each example is comprised of 'id','context','question' and list of 'text'(answers)
- 2) Create data generator by python yield way,the labels should be got by sub string searching function.
- 3) Build a customized model by adding a bidirectional GRU layer and classification layer on top of the bert pretrained model
- 4) Build customized loss function and evaluation function, the parameters of the input should be y_true and y_pred.
- 5) Build customized callback class inherited from keras.callbacks.Callback and redefine on_epoch_end function with epoch as input, the model will be saved once better metric like 'F1' is shown.
- 6) Train the model with model.fit_generator function.
- 7) Build predict_to_file function with functions from **official evaluate.py** to output the answers of all examples in json format.(f.write(json.dumps(R)))
## Special Code
```python
# load data
def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    return D
# search answer
def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
# MaskedSoftmax Layer
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
# bidirectional gru
output = Bidirectional(GRU(384,return_sequences=True))(model.output) 
# customized loss and accuracy function
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
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(learing_rate),
    metrics=[sparse_accuracy]
)
# predict to file 
def predict_to_file(infile, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1])
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()
# callback function
class Evaluator(keras.callbacks.Callback):
    """
    评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(
            os.path.join(data_dir,'dev.json')
            # os.path.join(data_dir,'demo_dev.json')
        )
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            model.save_weights(os.path.join(output_dir,'roberta_best_model.weights'))
            model.save(os.path.join(output_dir,'roberta_best_model.h5'))
        metrics['BEST_F1'] = self.best_val_f1
        print(metrics)
evaluator = Evaluator()
# model training
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[evaluator]
)
# find longest common sequences
def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    p = i+1
    return s1[p-max_len:p], max_len
# calculate F1 score
def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
#         if args.debug:
#             print(json.dumps(ans_segs, ensure_ascii=False))
#             print(json.dumps(prediction_segs, ensure_ascii=False))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0*lcs_len/len(prediction_segs)
        rec = 1.0*lcs_len/len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)
# calculate EM score
def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em
```