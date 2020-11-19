## Bayesian sentiment analysis
Build a classic **Bayesian** model to deal with text classification task, you can learn the basic steps to solve text classification problem by traditional ML way.

### Packages
- pandas
- os
- jieba
- sklearn
- tqdm
- logging

### Important functions
- pandas.read_csv(path,sep='\t',names)
- sklearn.feature_extraction.text.TfidfVectorizer
- sklearn.naive_bayes.MultinomialNB()
- sklearn.feature_selection.SelectKBest
- sklearn.feature_selection.chi2
- sklearn.metrics.classification_report
- sklearn.metrics.confusion_matrix

### Main process
- read and preprocess data
- extract TF-IDF feature
- select core features
- build and train Bayesian model
- evaluate the trained model 

### Dataset
You can get the data from the [link](https://github.com/bojone/bert4keras/tree/master/examples/datasets), the dataset is divided into three parts:sentiment.train.data,sentiment.valid.data,sentiment.test.data

### Run
You can just run this program step by step and get a complete understanding as how to do text classification by traditional ML way.

### Special code
```python
# the commonly used following code is mainly used to create the logger
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
```