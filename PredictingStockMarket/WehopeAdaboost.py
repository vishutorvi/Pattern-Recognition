# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:42:23 2017

@author: vishw
"""
from __future__ import print_function

from pprint import pprint
from time import time
import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import os

#print(__doc__)
scaler = MinMaxScaler(feature_range=(0,1))
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if 1:
    Test2 = 'C:\\TermProject\\Classified\\20122\\'
    Test1 = 'C:\\TermProject\\Classified\\20121\\'
    Test0 = 'C:\\TermProject\\Classified\\20120\\'
    AllTest = [[Test2, '2'], [Test1, '1'], [Test0, '0']]
    
    
    Train2 = 'C:\\TermProject\\Classified\\20112\\'
    Train1 = 'C:\\TermProject\\Classified\\20111\\'
    Train0 = 'C:\\TermProject\\Classified\\20110\\'
    AllTrain = [[Train2, '2'], [Train1, '1'], [Train0, '0']]
    
    dirWalk = AllTrain + AllTest
    data = []
    for aDir, sType in dirWalk:
        for fDir in os.walk(aDir):
            for ffDir in fDir[1:]:
                for fffDir in ffDir:
                    file = open(aDir + fffDir, 'r')
                    data += [[file.read(), sType]]
                    file.close()
data = np.array(data)
#print(data)
data1 = []
data2 = []
for i in range(len(data)):
    if i%2==0:
        data2 += [data[i]]
    else:
        data1 += [data[i]]
d1 = open('data1.txt','w')
data1 = np.array(data1)
#print(data1)
d1.write(str(data1))
d1.close()
d2 = open('data2.txt','w')
data2 = np.array(data2)
d2.write(str(data2))
d2.close()
#print(data1[:,0])
#print(data1[:,1])

if 1:
    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', AdaBoostClassifier()),
        #('clf', RandomForestClassifier()),
    ])
    
    be = ()
    #for pen in ('elasticnet'):
    pen = 'elasticnet'
    #for niter in (10, 40, 100):
    niter = 40
    #for alph in (0.001,):#0.1, 0.001, 
    alph = 0.001
    #class_weights = list(np.ones(len(data1[:,1]),dtype='float64')/len(data1[:,1]))
    be += (SGDClassifier(max_iter=niter,alpha=alph,penalty=pen),)
    
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.25),
        'vect__max_features': (10000,40000),
        'vect__ngram_range': ((1, 1),(1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (False,True),
        'tfidf__norm': ('l2','l1'),
        'clf__max_depth':(5,10),
        'clf__n_estimators':(10,20,60,240,640,1600),
        'clf__algorithm':('SAMME',),
        'clf__base_estimator':be,
        #'clf__learning_rate':(0.01,0.1,1),
        #'clf__learning_rate':(.001,.1,1,10),
    }
    
    if __name__ == "__main__":
        # multiprocessing requires the fork to happen in a __main__ protected
        # block
    
        # find the best parameters for both the feature extraction and the
        # classifier
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)
    
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = time()
        #scaler.fit(data1[:,0])
        grid_search.fit(data1[:,0], data1[:,1].astype(int))
        print("done in %0.3fs" % (time() - t0))
        print()
        
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        prediction = grid_search.best_estimator_.predict(data2[:,0])
        #grid_search1 = GridSearchCV(pipeline, best_parameters, n_jobs=-1, verbose=1)
        #grid_search1.fit(data1[:,0], data1[:,1])
        #prediction = grid_search1.predict(data2[:,0])
        print('predictions = ',np.sum(prediction == data2[:,1].astype(int))/len(data2))