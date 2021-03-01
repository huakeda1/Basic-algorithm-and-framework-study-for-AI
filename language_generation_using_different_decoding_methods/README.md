## Language Generating Method
There are mainly three decoding methods(**greedy search,beam search, sampling search**) for language generation with Transformers, we try to realize these methods by simple numpy way so as to have deep understanding of these methods, we also use these internal integrated methods to generate text with Transformers(GPT2)
### Greedy Search
Greedy search simply selects the word with **the highest probability** as its next word  
### Beam Search
Beam search reduces the risk of missing hidden high probability word sequences by keeping the most likely num_beams of hypotheses at each time step and eventually choosing the hypothesis that has the overall highest probability.  
Beam search will always find an output sequence with higher probability than greedy search, but is not guaranteed to find the most likely output. 
The most common **n-grams penalty** makes sure that no n-gram appears twice by manually setting the probability of next words that could create an already seen n-gram to 0.  
### Sampling Search
In its most basic form, sampling means randomly picking the next word according to its conditional probability distribution.
It becomes obvious that language generation using sampling is not deterministic anymore. 
That is the big problem when sampling word sequences: the models often generate incoherent gibberish, a trick is to make the distribution sharper (increasing the likelihood of high probability words and decreasing the likelihood of low probability words) by **lowering the so-called temperature of the softmax**. 
#### Top-K Sampling
In Top-K sampling, the K most likely next words are filtered and the probability mass is redistributed among only those K next words.    
#### Top-P Sampling
Instead of sampling only from the most likely K words, in Top-p sampling chooses from **the smallest possible set of words whose cumulative probability exceeds the probability p**. The probability mass is then redistributed among this set of words. This way, the size of the set of words (a.k.a the number of words in the set) can dynamically increase and decrease according to the next word's probability distribution