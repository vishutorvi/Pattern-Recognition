# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:00:26 2017

@author: Dell PC
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats

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

#One against rest     
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
    a12 = ssperceptrononeagainstother(X_norm[Y==1], X_norm[Y==2], a0, eta)
    a23 = ssperceptrononeagainstother(X_norm[Y==2], X_norm[Y==3], a0, eta)
    a13 = ssperceptrononeagainstother(X_norm[Y==1], X_norm[Y==3], a0, eta)

    #correct classification count
    correct = 0
    
    #Augment Test data
    I = np.ones((len(X_test),1))
    X_test_norm = np.append(I,np.array(X_test_norm), axis=1)
    
    class_1=np.zeros((1,3))
    for j in range(len(Y_test)):
        if np.dot(a12,X_test_norm[j,:]) > 0:
            class_1[0,0] = 1
        else:
            class_1[0,0] = 2
        
        if np.dot(a13,X_test_norm[j,:]) > 0:
            class_1[0,1] = 1
        else:
            class_1[0,1] = 3
            
        if np.dot(a23,X_test_norm[j,:]) > 0:
            class_1[0,2] = 2
        else:
            class_1[0,2] = 3
        
        mode, count = stats.mode(class_1[0])
        if count==1:
            print('Ambiguous sample=',j,Y_test[j])
        elif Y_test[j] == mode[0]:
            correct += 1
            print('correctly classified:',j,Y_test[j],mode)
        elif Y_test[j] != mode[0]:
            print('misclassified:',j,Y_test[j],mode)
    
    print('\n----------------Accuracy for one against rest-----------------')        
    print('--------------------------------------------------------------')
    print('Classification Accuracy=',correct/len(X_test_norm))
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
        a[i,:] = singlePerceptron(X_norm, Y, i+1, a0, eta)
    
    #misclassification
    correct = 0
    
    #ambiguous count
    l = 0
    
    #Augment Test data
    I = np.ones((len(X_test),1))
    X_test_norm = np.append(I,np.array(X_test_norm), axis=1)

    #classification result for test data
    for j in range(len(Y_test)):
        #print(j)
        l = 0
        class_1 = ''
        for k in range(len(a)):
            if(np.dot(a[k,:],X_test_norm[j,:]) > 0):
                l += 1
                class_1 = k+1

        if(Y_test[j] == class_1 and l == 1):
            correct += 1
            print('yes=\t', j, Y_test[j], class_1)
        elif(Y_test[j] != class_1 and l == 1):
            print('no=\t', j, Y_test[j], class_1)
        else:    
            print('ambiguous=\t', j, Y_test[j])
    print('\n----------------Accuracy for one against rest-----------------')        
    print('--------------------------------------------------------------')
    print('Classification Accuracy=',correct/len(X_test_norm))
    print('--------------------------------------------------------------')
        
singlesampleperceptron_one_against_rest('./wine_train.txt','./wine_test.txt', 13)
#singlesampleperceptron_one_against_other('./wine_train.txt','./wine_test.txt', 13)