# Pairwise-Word-Interaction-Modeling-by-Tensorflow-1.0
The model is based on paper[1]. I read this paper and decide to made it by myself. This paper has in this project.

[1]He H, Lin J. Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement[C]// Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016:937-948.

I choose Tensorflow r1.0 to write my code. website:https://www.tensorflow.org/

Prｅtrain word2vec: 
I used https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec and my train set is common text8, but I remove stop-words.

I save word2vec in two pkl file: dictionary.pkl and embeddings.pkl. Both dictionary.pkl and embeddings.pkl are python dictionary. dictionary.pkl, its key is word and value is index of word. embeddings.pkl, its key is index of word and value is the vetor of correspnding word. These two files used in model.py and read.py. 
Unfortunately, these files is too big to upload to github, instead put it on, I turn to submit my script.

----------------------------------------------------------------------------------------------------------------------------------
*Updated:There are some problem in this codes. And I haven't finished it.
