
# Pairwise-Word-Interaction-Modeling-by-Tensorflow-1.0
The model is based on 

* He H, Lin J. Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement[C]// Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016:937-948.

The paper is available [there](http://www.aclweb.org/anthology/N/N16/N16-1108.pdf).I studied this paper and implement its codes by myself. I chose [Tensorflow r1.0](https://www.tensorflow.org/) to write my code.<br>

pretrained word2vec: I used [word2vec](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec) and my train set is the common text8, but I removed stop-words.<br>

I save word2vec in two pkl file: dictionary.pkl and embeddings.pkl. Both dictionary.pkl and embeddings.pkl are python dictionary.

dictionary.pkl, its key is word and value is index of word.<br> 
embeddings.pkl, its key is index of word and value is the vetor of correspnding word.<br> 

These two files are used in model.py and read.py. Unfortunately, these files are too big to upload to github. Instead, I turn to submit my script.

**Updated:There are some problem in these codes. And I haven't finished it.**
