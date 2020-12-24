import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config(object):
    def __init__(self,):
        self.maxlen = 256
        self.epochs = 2
        self.batch_size = 16
        self.learing_rate = 2e-5

        self.data_dir=os.path.join(ROOT,'data')
        self.output_dir = os.path.join(ROOT,'output')

        self.bert_dir = os.path.join(ROOT,'pretrained_model/chinese-roberta-wwm-ext')
        self.config_path = f'{self.bert_dir}/bert_config.json'
        self.checkpoint_path = f'{self.bert_dir}/bert_model.ckpt'
        self.dict_path = f'{self.bert_dir}/vocab.txt'