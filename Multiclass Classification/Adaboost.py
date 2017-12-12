# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:29:52 2017

@author: vishw
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats
import itertools as it
from bisect import bisect_right
import math
import random

def gradientdescent_batch(Y, a0, eta, m):
    a = a0
    miss = 1
    l = 1
    #print(Y)
    x = np.zeros((1,len(Y[0])))
    while(miss==1 and l < 5000):
        miss = 0
        for i in range(len(Y)):
            if(np.dot(a,Y[i,:]) <= m):
                x = x + ((m-np.dot(a,Y[i,:]))*Y[i,:])/(np.linalg.norm(Y[i,:])**2)#eta*Y[i,:]
                miss=1
        l += 1
        a = a + eta*x
    return a



def ssperceptrononeagainstother(X, Y, a, eta, b):
    #number of samples
    Ix = np.ones((len(X),1))
    Iy = np.ones((len(Y),1))
    
    #print(X)
    #print(Y)
    
    #augmented matrix add 1, strip the class information
    augX = np.append(Ix, X, axis = 1)
    augY = np.append(Iy, Y, axis = 1)
    
    #negate Y
    augY = augY*(-1)
    
    #add them to a single matrix
    #print(augX)
    #print(augY)
    augMat = np.concatenate((augX,augY), axis = 0)
    a = gradientdescent_batch(augMat, a, eta, b)
    return a

def find_le(a, x):
    #print(a)
    #print(x)
    'Find rightmost value less than or equal to x'
    i = bisect_right(a, x)
    #print(i)
    return i

def errorCal(Y_test,Y_unique,trained,Ypairs,X_test_norm,weight):
    class_1=[]
    I = np.ones((len(X_test_norm),1))
    X_test_norm = np.append(I,np.array(X_test_norm), axis=1)
#    print(trained)
    #print(X_test_norm)
    for i in range(len(Y_test)):
        class_2 = {}
        for y in Y_unique:
            class_2[y] = 0
        for j in range(len(trained)):
            class_3=np.zeros((len(Y_unique)))
            for n, aClass, bClass in Ypairs: 
               # print(len(trained[j,0]))
                #print(len(X_test_norm[i,:]))
                if np.dot(trained[j,n],X_test_norm[i,:]) > 0:
                    class_3[n] = aClass
                else:
                    class_3[n] = bClass
            if int(stats.mode(class_3)[1]) >= (len(Y_unique) // 2) + 1:
                #print(stats.mode(class_3)[0])
                #print(weight[j])
                class_2[int(stats.mode(class_3)[0])] += weight[j]
        #print(class_2)
        mClss = 0
        bestClss = -1
        for clss in class_2:
            if class_2[clss] > mClss:
                mClss = class_2[clss]
                bestClss = clss
            elif class_2[clss] == mClss:
                bestClss = -1
        class_1 += [bestClss]
    
    class_1 = np.array(class_1)    
    error = (np.sum(class_1 != Y_test))/len(Y_test)
    return error, class_1

def boost(ftrain, ftest, fcount, eta, b):
    features = ["w" + str(x) for x in range(1,fcount+1)]
    features = ['Target']+features
    df = pd.read_csv(ftrain, names = features, sep=',')

    #margin
    df_test = pd.read_csv(ftest,names = features, sep = ',')
    X = df[features[1:]]
    Y = df[features[0]]
    
    #Test data
    X_test = df_test[features[1:]]
    Y_test = df_test[features[0]]

    #Unique
    Y_unique = np.unique(np.array(Y_test))
    Yp = list(it.combinations(Y_unique,2))
    
    Ypairs = []
    n = 0
    for a1, b1 in Yp:
        Ypairs += [[n, a1, b1]]
        n += 1
        
    #normalizing data
    X_norm = preprocessing.scale(X,axis=0,with_mean=True,with_std=True)
    X_test_norm = preprocessing.scale(X_test,axis=0,with_mean=True,with_std=True)
    
    #initial weights
    a0 = np.ones((len(X_norm[0])+1))

    #initial sample weights 
    wi = np.ones(len(X_norm))/float(len(X_norm))
    
#    print(wi)
#    print(len(X_norm))
    #constant samples
    ratio = len(X_norm)//4
    #trained weight vector
    trained = []
    weight = []
    for _ in range(10):
        X_normSampled = []
        classifiers = []
        Y_Sampled = []
        wics = list(np.cumsum(wi))
        if wics[-1] < 1.0:
            wics[-1] = 1.0
#        print(wics)
        for i in range(ratio):
            sample = find_le(wics, np.random.ranf())
            #print(sample)
            X_normSampled += [X_norm[int(sample)]]
            Y_Sampled += [Y[sample]]
        for i in Y_unique:
            X_normSampled += random.sample(list(X_norm[Y==i]),1)
            Y_Sampled += [i]
        
        X_normSampled = np.array(X_normSampled)
        for n, aClass, bClass in Ypairs:
            classifiers += [ssperceptrononeagainstother(X_normSampled[Y_Sampled==aClass], X_normSampled[Y_Sampled==bClass], a0, eta, b)]
        
        sampleError, sampredictions = errorCal(np.array(Y),Y_unique,np.array([classifiers]),Ypairs,X_norm,[1])
        if  sampleError == 0:
            alpha = 0
        else:
            alpha = (1/2)*math.log(((1-sampleError))/sampleError)
        
        for y in range(len(Y)):
            if sampredictions[y] != Y[y]:
                wi[y] *= math.exp(alpha)
            else:
                wi[y] *= math.exp(-alpha)
        
        zk = np.sum(wi)
        wi = wi/zk
        trained +=[classifiers]
        weight += [alpha]
  
    Terror, Tpredictions = errorCal(Y_test,Y_unique,np.array(trained),Ypairs,X_test_norm,weight)
    print(Tpredictions)
    print('-----------------Adaboost Digit Data Set------------------------')
    print('Accuracy=',100.0-Terror)
    print('---------------------------------------------------------------')
    
#boost('./wine_train.txt','./wine_test.txt', 13, .5, 5)
boost('./digit_train.txt','./digit_test.txt', 256, 0.05, -1)