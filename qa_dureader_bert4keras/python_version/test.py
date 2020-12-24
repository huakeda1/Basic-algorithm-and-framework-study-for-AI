#!/usr/bin/env python
# -*- coding: utf-8 -*-
from train import MaskedSoftmax,sparse_accuracy,evaluate,predict_to_file
from keras.models import load_model
from utils.config import Config
from bert4keras.tokenizers import Tokenizer
import os

if __name__=="__main__":
    args=Config()
    tokenizer = Tokenizer(args.dict_path, do_lower_case=True)
    model=load_model(os.path.join(args.output_dir,'roberta_best_model.h5'),custom_objects={'MaskedSoftmax':MaskedSoftmax,'sparse_accuracy':sparse_accuracy})
    print(evaluate(os.path.join(args.data_dir,'dev.json'),model,tokenizer,args.maxlen))
    predict_to_file(os.path.join(args.data_dir,'dev.json'), os.path.join(args.output_dir,'dev.pred1.json'),model,tokenizer,args.maxlen)
    # predict_to_file(os.path.join(args.data_dir,'test2.json'),  os.path.join(args.output_dir,'pred2.json'),model,tokenizer)