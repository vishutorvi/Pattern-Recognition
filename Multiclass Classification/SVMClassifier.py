# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 01:05:24 2017

@author: vishw
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing

def datasetForSpecificClass(X, class1, class2):
    data1 = X[X[:,0]==class1]
    data2 = X[X[:,0]==class2]
    data = np.append(data1,data2, axis = 0)
    return data

def svmClassify(ftrain, ftest, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]
    features = ['Target']+features
    df_main = pd.read_csv(ftrain, names = features, sep=',')

    df_main_test = pd.read_csv(ftest,names = features, sep = ',')
    
    #ran = [[1,2],[2,3],[1,3]]
    ran = [[0,1],[1,2],[0,2]]
    for i, j in ran:
        df = datasetForSpecificClass(np.array(df_main), i, j)
        df_test = datasetForSpecificClass(np.array(df_main_test), i, j)
        
        #print(df)
        X = df[:,1:]
        Y = df[:,0]
        
        #Test data
        X_test = df_test[:,1:]
        Y_test = df_test[:,0]
    
        #normalizing data
        X_norm = preprocessing.scale(X,axis=0,with_mean=True,with_std=True)
        X_test_norm = preprocessing.scale(X_test,axis=0,with_mean=True,with_std=True)
        
        clf = svm.SVC()
        clf.fit(X_norm,Y)
        
        predictions = clf.predict(X_test_norm)
        print('SVM Classification on',i,'-',j,'class accuracy=',(np.sum(predictions == Y_test)/len(X_test_norm))*100,'%')
    
    
#svmClassify('./wine_train.txt','./wine_test.txt', 13)
svmClassify('./digit_train.txt','./digit_test.txt', 256)