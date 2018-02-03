# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:28:41 2017

@author: Dell PC
"""
from __future__ import print_function
import json
import nltk
import numpy as np
import pandas as pd
import sklearn.feature_extraction as sk
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from operator import itemgetter
import os
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
stemmer = SnowballStemmer('english')
#transformer = TfidfTransformer()

df = pd.read_csv('C:/TermProject/ClassifiedRnn/ALL.txt',sep='\t',header=None)
print(df.head())
nparr = np.array(df)
#aFile = open("filename.txt",'r+')
if 1:
    for i in range(len(nparr)):
        sentence = nparr[i][1]
        tokens = nltk.word_tokenize(sentence)
        stokens = []
        for token in tokens:
            stokens += [stemmer.stem(token)]
        tagged = nltk.pos_tag(stokens)
        tsentence = ''
        for word in tagged:
            if word[1] in ['DT','JJ','JJR','JJS','NN','MD','NNS','PDT','RB','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ']:
                if word[0] not in ['the','a'] and len(word[0])>1:
                    tsentence += word[0]+" "
        nparr[i][1] = tsentence
    cls = nparr[:,0]
#        if i in [390]:
#            print(nparr[i][0])

if 1:  
    vec = sk.text.CountVectorizer(ngram_range=(1, 2))
    pos_vectorized = vec.fit_transform(nparr[:,1])
    
    colSum = []
    
    pos_vectorized = transformer.fit_transform(pos_vectorized).toarray()
    #pos_vectorized = pos_vectorized.toarray()
    for i in range(len(pos_vectorized[0])):
        colSum += [[sum(pos_vectorized[:,i]), i]]
    
    colSum.sort(key=itemgetter(0),reverse = True)
    
    colSum = np.array(colSum[:10000])
    print(colSum)
    pos_vectorizedNew = []
    for i in range(len(pos_vectorized)):
        newRow = []
        oldRow = pos_vectorized[i]
        for j in colSum[:,1]:
            newRow += [oldRow[int(j)]]
        pos_vectorizedNew += [newRow]
    pos_vectorized = np.array(pos_vectorizedNew)
        
    print(pos_vectorized)

if 0:
    train = "C:/TermProject/ClassifiedRnn/trainREC.csv"
    test = "C:/TermProject/ClassifiedRnn/testREC.csv"
    train = open(train, 'w')
    test = open(test, 'w')
    
    for i in range(len(pos_vectorized)):
        day = pos_vectorized[i]
        string = str(cls[i])
        for word in day:
            string += ',' + str(word)
        if i > len(pos_vectorized) // 2:
            train.write(string + '\n')
        else:
            test.write(string + '\n')
    
    train.close()
    test.close()