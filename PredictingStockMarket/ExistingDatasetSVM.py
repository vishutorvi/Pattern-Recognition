# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:42:23 2017

@author: vishw
"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pprint
from time import time
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import string

print(__doc__)

def wordtokenize(text):
    sentence = ''
    for x in text.split(" "):
        sentence += x.lower()+' '
    return sentence
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if 1:
    dataset = pd.read_csv('./Combined_News_DJIA.csv', header=0, sep=',')
    data = np.array(dataset)[:,2:]
    finaldata = []
    for i in range(0,len(data)):
        datawords = ''
        for j in range(0,len(data[0])):
            if type(data[i,j]) == float:
                continue
            value = data[i,j].translate(str.maketrans('','',string.punctuation))#.lower()
            datawords += wordtokenize(value)
        finaldata += [datawords] 
    label = np.array(dataset)[:,1]
    
data = np.array(finaldata)
print(data)
data1 = []
data2 = []
label1 = []
label2 = []
for i in range(len(data)):
    if i%2==0:
        data2 += [data[i]]
        label2 += [label[i]]
    else:
        data1 += [data[i]]
        label1 += [label[i]]
#d1 = open('data1.txt','w')
data1 = np.array(data1)
label1 = np.array(label1)
#print(data1)
#d1.write(str(data1))
#d1.close()
#d2 = open('data2.txt','w')
data2 = np.array(data2)
label2 = np.array(label2)
#d2.write(str(data2))
#d2.close()
#print(data1[:,0])
#print(data1[:,1])

if 1:
    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])
    
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 50000, 100000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.1, 0.01),#loss function:'squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 80),
    }
    
    if __name__ == "__main__":
        # multiprocessing requires the fork to happen in a __main__ protected
        # block
    
        # find the best parameters for both the feature extraction and the
        # classifier
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = time()
        grid_search.fit(data1, label1.astype(int))
        print("done in %0.3fs" % (time() - t0))
        print()
        
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        prediction = grid_search.best_estimator_.predict(data2[:])
        #grid_search1 = GridSearchCV(pipeline, best_parameters, n_jobs=-1, verbose=1)
        #grid_search1.fit(data1[:,0], data1[:,1])
        #prediction = grid_search1.predict(data2[:,0])
        print('predictions = ',np.sum(prediction == label2.astype(int))/len(data2))