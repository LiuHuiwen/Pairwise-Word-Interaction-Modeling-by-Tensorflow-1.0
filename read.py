# -*- coding: utf-8 -*-
import numpy as np
import pickle

read_data = open('../W2V/dictionary.pkl', 'rb')
dictionary = pickle.load(read_data)
read_data.close()

class ReadDataset:

    def __init__(self, filename, len_of_sen=32):
        self.len_of_sen = len_of_sen

        f = open(filename, 'rb')
        self.dataset = f.readlines() #read dataset to class ReadDataset
        f.close()

        self.data_index = 1
        self.lines = 9840

    def next_batch(self, batch):
        x = [] #sentence one
        y = [] #sentence two 
        z = [] #label
        for i in range(batch):
            data = self.dataset[self.data_index].split('\t')
            s1 = self.sentence_to_index(data[1]) #return a list of index
            s2 = self.sentence_to_index(data[2])
            l = float(data[4])
            x.append(s1)
            y.append(s2)
            z.append(l)
            self.data_index = self.data_index%self.lines + 1
        
        return x,y,z # stence label

    def sentence_to_index(self,stc_str):#stc:sentence, str: string (dtype)
        global dictionary # key: word, value: index in embedding
        index_lst = [] #return, list of index represent sentence
        padding = dictionary['UNK']
        word_lst = stc_str.split()
        for word in word_lst:
            if(dictionary.has_key(word.lower())):
                index_lst.append(dictionary[word.lower()])
            else:
                index_lst.append(padding)
        lenght = len(index_lst)
        if(lenght>=self.len_of_sen):
            return index_lst[:self.len_of_sen]
        else:
            tmp = self.len_of_sen - lenght
            index_lst.extend([padding for i in range(tmp)])
            return index_lst



if __name__ == '__main__':
    f = ReadDataset('txt', 18)
    x, y, z = f.next_batch(2)
    print x
    print y
    print z

