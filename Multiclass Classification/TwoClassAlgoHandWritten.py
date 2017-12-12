# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:48:32 2017

@author: Dell PC
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

def gradientdescent(Y, a0, eta):
    a = a0;
    miss = 1
    while(miss==1):
        miss = 0
        for i in range(len(Y)):
            if(np.dot(a,Y[i,:]) <= 0):
                a = a + eta*Y[i,:]
                miss=1
    return a
  
def ssperceptrononeagainstother(X, Y, a, eta):
    #number of samples
    Ix = np.ones((len(X),1))
    Iy = np.ones((len(Y),1))
    
    #augmented matrix add 1, strip the class information
    augX = np.append(Ix, X, axis = 1)
    #print(augX)
    augY = np.append(Iy, Y, axis = 1)
    #print(augY)
    
    #negate Y
    augY = augY*(-1)
    
    #add them to a single matrix
    augMat = np.concatenate((augX,augY), axis = 0)
    
    #print(len(augMat))
    
    a = gradientdescent(augMat, a, eta)
    
    return a

      
def singlePerceptron(X, label, c, a0, eta):
    #value to augment
    I = np.ones((len(X),1))
    #Augment Vector
    Y = np.append(I,np.array(X), axis=1)
    #Negate which does not belong to class c
    for i in range(len(Y)):
        if label[i] != c:
            #print('not c=',i)
            Y[i,:] = Y[i,:] * (-1)
    
    #Gradient descent call
    a = gradientdescent(Y, a0, eta)
    return a
   
def singlesampleperceptron_one_against_other(ftrain, ftest, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]
    features = ['Target']+features
    df = pd.read_csv(ftrain, names = features, sep=',')

    df_test = pd.read_csv(ftest,names = features, sep = ',')
    X = df[features[1:]]
    Y = df[features[0]]
    eta = 0.6
    
    #Test data
    X_test = df_test[features[1:]]
    Y_test = df_test[features[0]]

    #normalizing data
    X_norm = preprocessing.scale(X,axis=0,with_mean=True,with_std=True)
    X_test_norm = preprocessing.scale(X_test,axis=0,with_mean=True,with_std=True)
    
    #initial weights
    a0 = np.ones((len(X_norm[0])+1))

    #trained weight vector
    a12 = ssperceptrononeagainstother(X_norm[Y==0], X_norm[Y==1], a0, eta)
    a23 = ssperceptrononeagainstother(X_norm[Y==1], X_norm[Y==2], a0, eta)
    a13 = ssperceptrononeagainstother(X_norm[Y==0], X_norm[Y==2], a0, eta)

    #Augment Test data
    I = np.ones((len(X_test),1))
    X_test_norm = np.append(I,np.array(X_test_norm), axis=1)
    
    correct12 = 0 
    total12count = 0
    correct13 = 0
    total13count = 0
    correct23 = 0
    total23count = 0
    
    for j in range(len(Y_test)):
        if(Y_test[j] == 0 or Y_test[j] == 1):
            total12count += 1
            if np.dot(a12,X_test_norm[j,:]) > 0:
                cls = 0
            else:
                cls = 1
            if Y_test[j]==cls:
                correct12 += 1
        
        if(Y_test[j] == 0 or Y_test[j] == 2):
            total13count += 1
            if np.dot(a13,X_test_norm[j,:]) > 0:
                cls = 0
            else:
                cls = 2
            if Y_test[j]==cls:
                correct13 += 1
        
        if(Y_test[j] == 1 or Y_test[j] == 2):
            total23count += 1
            if np.dot(a23,X_test_norm[j,:]) > 0:
                cls = 1
            else:
                cls = 2
            if Y_test[j]==cls:
                correct23 += 1
        
    p12 = correct12/float(total12count) * 100
    p13 = correct13/float(total13count) * 100
    p23 = correct23/float(total23count) * 100
    print('\n----------------Accuracy for one against rest-----------------')        
    print('--------------------------------------------------------------')
    print('The performance of class 1-2  classifier on handwritten digit data set is =',p12);
    print('The performance of class 1-3  classifier on handwritten digit data set is =',p13);
    print('The performance of class 2-3  classifier on handwritten digit data set is =',p23);
    print('--------------------------------------------------------------')
    
    
    
def singlesampleperceptron_one_against_rest(ftrain, ftest, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]
    features = ['Target']+features
    df = pd.read_csv(ftrain, names = features, sep=',')

    df_test = pd.read_csv(ftest,names = features, sep = ',')
    X = df[features[1:]]
    Y = df[features[0]]
    eta = 0.6
    
    #Test data
    X_test = df_test[features[1:]]
    Y_test = df_test[features[0]]

    #normalizing data
    X_norm = preprocessing.scale(X,axis=0,with_mean=True,with_std=True)
    X_test_norm = preprocessing.scale(X_test,axis=0,with_mean=True,with_std=True)
    
    #initial weights
    a0 = np.ones((len(X_norm[0])+1))

    #trained weight vector
    a = np.zeros((len(np.unique(Y)),(len(X_norm[0])+1)))
        
    #weights updating
    for i in range(len(np.unique(Y))):
        print('i=',i)
        a[i,:] = singlePerceptron(X_norm, Y, i, a0, eta)
    
    #Augment Test data
    I = np.ones((len(X_test),1))
    X_test_norm = np.append(I,np.array(X_test_norm), axis=1)
    
    print(a[0])
    #classification count
    correct1 = 0
    correct2 = 0
    correct3 = 0
    #classification result for test data
    for j in range(len(Y_test)):
        # for class 1
        if (np.dot(a[0,:],X_test_norm[j,:])> 0):
            cls = 1
        else:
            cls = -1
        
        if ((Y_test[j] == 0 and cls == 1) or ((Y_test[j] == 1) or Y_test[j] == 2) and cls == -1):
            correct1 += 1
        
        # for class 2
        if (np.dot(a[1,:],X_test_norm[j,:])> 0):
            cls = 1
        else:
            cls = -1
        
        if ((Y_test[j] == 1 and cls == 1) or ((Y_test[j] == 0) or Y_test[j] == 2) and cls == -1):
            correct2 += 1
        
        # for class 3
        if (np.dot(a[2,:],X_test_norm[j,:])> 0):
            cls = 1
        else:
            cls = -1
        
        if ((Y_test[j] == 2 and cls == 1) or ((Y_test[j] == 0) or Y_test[j] == 1) and cls == -1):
            correct3 += 1
    
    p1 = correct1/float(len(X_test)) * 100
    p2 = correct2/float(len(X_test)) * 100
    p3 = correct3/float(len(X_test)) * 100        
    print('\n----------------Accuracy for one against rest-----------------')        
    print('--------------------------------------------------------------')
    print('The performance of two-class classifier for class 1 against the rest using single sample on wine data set is =',p1);
    print('The performance of two-class classifier for class 2 against the rest using single sample on wine data set is =',p2);
    print('The performance of two-class classifier for class 3 against the rest using single sample on wine data set is =',p3);
    print('--------------------------------------------------------------')
        
singlesampleperceptron_one_against_rest('./digit_train.txt','./digit_test.txt', 256)
#singlesampleperceptron_one_against_other('./digit_train.txt','./digit_test.txt', 256)