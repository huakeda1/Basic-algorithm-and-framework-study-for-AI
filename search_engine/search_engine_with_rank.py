#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce
import numpy as np
import os
import re
from scipy.spatial.distance import cosine


# In[2]:


csv_file='dataset/news.csv'
if os.path.exists(csv_file):
    news=pd.read_csv(csv_file,encoding='gb18030',nrows=20000)
    news['content']=news['content'].fillna('')
    news['cut_words']=news['content'].apply(lambda x:' '.join(list(jieba.cut(x))))
    news['cut_words'].to_csv('dataset/news_content.csv')
    print('news csv has been successfully processed')


# In[3]:


def reduce_and(vectors):
    return reduce(lambda a,b:a&b,vectors)


# In[4]:


class RetrievalEngine:
    def __init__(self,corpus):
        # token_pattern is set to be r"(?u)\b\w\w+\b" by default which can only accept words longer than two.
        # token_pattern is set to be r"(?u)\b\w+\b" which can accept single word or alpha.
        # vocabulary can give words which will be used to build matrix
        # max_df can filter words which have higher exist frequency in all docs
        # tf is decided only by current doc, tf equals frequency in single doc.
        # idf is decided by how many docs have this word and how many docs are given here.
        # idf equals to 1+log((total_docs)/(docs_contain_thisword)) or 1+log((1+total_docs)/(1+docs_contain_thisword))
        # tfidf means tf*idf.
        self.vectorizer=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_df=1.0,stop_words=[],vocabulary=None,use_idf=True,smooth_idf=True)
        self.vectorizer.fit(corpus)
        self.corpus=corpus
        self.d2w=self.vectorizer.transform(corpus).toarray()
        self.w2d=self.d2w.transpose()
    def get_words_id(self,words):
        ids=[self.vectorizer.vocabulary_[w] for w in words if w in self.vectorizer.vocabulary_]
        return ids
    def get_w2d_vectors(self,words):
        vectors=self.w2d[self.get_words_id(words)]
        return vectors
    # get the idnexes of docs which have all the specific words
    def get_combined_common_indices(self,words):
        try:
            indices=reduce_and([set(np.where(v)[0]) for v in self.get_w2d_vectors(words)])
            return indices
        except Exception as e:
            return []
    def get_sorted_indices(self,words):
        indices=self.get_combined_common_indices(words)
        query_vector=self.vectorizer.transform(words).toarray()[0]
        sorted_indices=sorted(indices,key=lambda indice:cosine(query_vector,self.d2w[indice]),reverse=True)
        return sorted_indices
    def get_requested_text(self,words):
        sorted_indices=self.get_sorted_indices(words)
        output=[self.corpus[indice] for indice in sorted_indices]
        return output
        


# In[5]:


corpus=[" ".join(list(jieba.cut("我爱吃香蕉")))," ".join(list(jieba.cut("你爱吃苹果")))," ".join(list(jieba.cut("苹果没有香蕉吃得好")))]
retrieval_engine=RetrievalEngine(corpus)
print(retrieval_engine.w2d)
print(retrieval_engine.vectorizer.vocabulary_)
words=list(jieba.cut("喜欢水果"))
print(retrieval_engine.get_words_id(words))

print(retrieval_engine.get_w2d_vectors(words))

print(retrieval_engine.get_combined_common_indices(words))
print(retrieval_engine.get_sorted_indices(words))
print(retrieval_engine.get_requested_text(words))


# In[ ]:




