#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import jieba
from tqdm import tqdm
def cut_sentence(data):
    new_data=data.apply(lambda x:' '.join(jieba.lcut(x)))
    return new_data
def load_stopwords(stop_words_dir):
    stopwords=[]
    with open(stop_words_dir,'r',encoding='utf-8') as f:
        for index,line in enumerate(tqdm(f.readlines())):
            if not line.strip():continue
            stopwords.append(line.strip())
    return stopwords


# In[2]:


import logging
import os
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
data_path = "./home/aistudio/data/sentiment"
log_path=os.path.join(data_path,'log.txt')
# 日志记录到文件
handler=logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# 日志打印到控制台
console=logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)
logger.info('The logger is successfully built for the model')


# In[3]:


data_path = "./home/aistudio/data/sentiment"
train_df=pd.read_csv(os.path.join(data_path,'sentiment.train.data'),sep='\t',names=["text","label"])
valid_df=pd.read_csv(os.path.join(data_path,'sentiment.valid.data'),sep='\t',names=["text","label"])
test_df=pd.read_csv(os.path.join(data_path,'sentiment.test.data'),sep='\t',names=["text","label"])

X_train=train_df['text']
y_train=train_df['label']

X_valid=valid_df['text']
y_valid=valid_df['label']

X_test=test_df['text']
y_test=test_df['label']

stop_words_dir=os.path.join(data_path,'stopwords.txt')
stopwords=load_stopwords(stop_words_dir)


# In[4]:


# get tfidf feature
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words=stopwords,max_features=5000,lowercase=False,sublinear_tf=True,max_df=0.8)
tfidf_vectorizer.fit(cut_sentence(X_train))
X_train_tfidf=tfidf_vectorizer.transform(cut_sentence(X_train))
X_valid_tfidf=tfidf_vectorizer.transform(cut_sentence(X_valid))
X_test_tfidf=tfidf_vectorizer.transform(cut_sentence(X_test))
print(X_train_tfidf.shape)
print(X_valid_tfidf.shape)
print(X_test_tfidf.shape)


# In[5]:


# feature selection
from sklearn.feature_selection import SelectKBest,chi2
selector=SelectKBest(chi2,k=3000)
X_train_tfidf_chi=selector.fit_transform(X_train_tfidf,y_train)
X_valid_tfidf_chi=selector.transform(X_valid_tfidf)
X_test_tfidf_chi=selector.transform(X_test_tfidf)
print(X_train_tfidf_chi.shape)
print(X_valid_tfidf_chi.shape)
print(X_test_tfidf_chi.shape)


# In[6]:


# Bayesian model
from sklearn.naive_bayes import MultinomialNB
# model evaluation
from sklearn.metrics import classification_report,confusion_matrix

classifier_nb=MultinomialNB(alpha=0.2)
classifier_nb.fit(X_train_tfidf,y_train)
y_train_pred=classifier_nb.predict(X_train_tfidf)
y_valid_pred=classifier_nb.predict(X_valid_tfidf)
y_test_pred=classifier_nb.predict(X_test_tfidf)

print(classifier_nb.score(X_test_tfidf,y_test))
print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))

classifier_nb.fit(X_train_tfidf_chi,y_train)
y_train_pred_chi=classifier_nb.predict(X_train_tfidf_chi)
y_valid_pred_chi=classifier_nb.predict(X_valid_tfidf_chi)
y_test_pred_chi=classifier_nb.predict(X_test_tfidf_chi)

print(classifier_nb.score(X_test_tfidf_chi,y_test))
print(classification_report(y_test,y_test_pred_chi))
print(confusion_matrix(y_test,y_test_pred_chi))


# ## Bayesian sentiment analysis
# Build a classic **Bayesian** model to deal with text classification task, you can learn the basic steps to solve text classification problem by traditional ML way.
# 
# ### Packages
# - pandas
# - os
# - jieba
# - sklearn
# - tqdm
# - logging
# 
# ### Important functions
# - pandas.read_csv(path,sep='\t',names)
# - sklearn.feature_extraction.text.TfidfVectorizer
# - sklearn.naive_bayes.MultinomialNB()
# - sklearn.feature_selection.SelectKBest
# - sklearn.feature_selection.chi2
# - sklearn.metrics.classification_report
# - sklearn.metrics.confusion_matrix
# 
# ### Main process
# - read and preprocess data
# - extract TF-IDF feature
# - select core features
# - build and train Bayesian model
# - evaluate the trained model 
# 
# ### Dataset
# You can get the data from the [link](https://github.com/bojone/bert4keras/tree/master/examples/datasets), the dataset is divided into three parts:sentiment.train.data,sentiment.valid.data,sentiment.test.data
# 
# ### Run
# You can just run this program step by step and get a complete understanding as how to do text classification by traditional ML way.
# 
# ### Special code
# ```python
# # the commonly used following code is mainly used to create the logger
# import logging
# import os
# logger=logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# data_path = "./home/aistudio/data/sentiment"
# log_path=os.path.join(data_path,'log.txt')
# # 日志记录到文件
# handler=logging.FileHandler(log_path)
# handler.setLevel(logging.INFO)
# formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# # 日志打印到控制台
# console=logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(formatter)
# 
# logger.addHandler(handler)
# logger.addHandler(console)
# logger.info('The logger is successfully built for the model')
# ```

# In[ ]:




