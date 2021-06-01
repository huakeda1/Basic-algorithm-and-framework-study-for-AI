# Search engine based on Tfidf model
We study the Principe of search engine based on Tfidf model,a Tfidf model is built with sklearn toolbox,the corpus is used to train the model, the words of the query text is firstly transformed to ids, then we try to find all the possible docs which contain all these words, this way can filter out most documents, on the other hand, these words will be transformed to one vector by the Tfidf model, then this vector is used to calculate cosine distance with vectors of all filtered docs, we can finally get the docs similar to the query text according to the sorted result.   

## Model Structure
The figure below shows the architecture of the search engine model:
![model_architecture](https://raw.github.com/huakeda1/Basic-algorithm-and-framework-study-for-AI/master/search_engine/associated_pngs/search_engine.png)  



