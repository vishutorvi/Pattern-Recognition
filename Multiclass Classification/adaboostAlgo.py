# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:53:45 2017

@author: Dell PC
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

def gradientdescent_batch(Y, a0, eta, m=1):
    a = a0
    miss = 1
    l = 1
    while(miss==1 and l < 5000):
        miss = 0
        x = np.zeros((1,len(Y[0])))
        for i in range(len(Y)):
            if(np.dot(a,Y[i,:]) <= m):
                x = x + ((m-np.dot(a,Y[i,:]))*Y[i,:]/(np.linalg.norm(Y[i,:])**2))#eta*Y[i,:]
                miss=1
        l += 1
        a = a + eta*x
    return a
  
def ssperceptrononeagainstother(X, Y, a, eta, b):
    #number of samples
    Ix = np.ones((len(X),1))
    Iy = np.ones((len(Y),1))
    
    #augmented matrix add 1, strip the class information
    augX = np.append(Ix, X[:,1:], axis = 1)
    augY = np.append(Iy, Y[:,1:], axis = 1)
    #negate Y
    augY = augY*(-1)
    
    #add them to a single matrix
    augMat = np.concatenate((augX,augY), axis = 0)
    a = gradientdescent_batch(augMat, a, eta, b)
    return a

def adaboost(x, y, s, c1, c2, eta, b, kmax):
    #weight vector , column
    W = np.array(np.ones((len(x),1))/len(x))
    y = np.array([y])
    #initialize weight vector with one, this is for perceptron
    a0 = np.ones((1, len(x[0])+1))
    #first add 1 to feature to make augmented vector
    Ix  = np.ones((len(x), 1))
    #add weight, add 1, 
    zx = np.append(y.T,W, axis = 1)
    zx = np.append(zx,Ix, axis = 1)
    zx = np.append(zx,x, axis = 1)
    j=1;
    z = []
    for i in range(len(x)):
        if (zx[i,0] == c1 or zx[i,0] == c2):
            z += [zx[i,:]]
            j = j + 1
    
    #new training set size
    m = len(z)
    
    #class predicted by classifier
    cls = 0
    
    alpha = np.zeros((1,kmax))
    hk = np.zeros((kmax,len(x[0])+1))
    #loop for AdaBoost
    for q in range(kmax):
        zs = np.array(z)
        zs = zs[zs[:,1].argsort()]
        zz = zs[0:s,:]
        b1=1
        b2=1
        z1 = []
        z2 = []
        for j in range(0,s):
            if (zz[j,0] == c1):
                z1 += [zz[j,:]]
                b1 = b1+1
    
            if (zz[j,0] == c2):
                z2 += [zz[j,:]]
                b2 = b2+1
        
        z1 = np.array(z1)
        z2 = np.array(z2)
        
        z11 = np.delete(z1, [1,2],1)#np.append([list(z1[:,0])], z1[:,3:],axis=0)
        z22 = np.delete(z2, [1,2],1)#np.append(z2[:,0], z2[:,3:],axis=1)#[z2[:,1], z2[:,4:]]
        hk[q,:] = ssperceptrononeagainstother(z11, z22, a0, eta, b)
        e = 0
        clss = []
        #loop through each test sample
        z = np.array(z)
        for i in range(m):
            #test only class 1 and class 2 samples
            if (np.dot(hk[q,:],z[i,2:]) > b):
                 cls = c1
                 clss += [c1]
            else:
                 cls = c2
                 clss += [c2]
                 
            if(int(z[i,0]) != int(cls)):   #incorrect
                e = e+1
        
        em = e/m
        if em == 0:
            alpha[0][q] = math.inf#0.5*np.log(0)
        else:
            alpha[0][q] = 0.5*np.log((1-em)/em)
        
        for i in range(m):
            if(np.dot(hk[q, :],z[i,2:]) > b):
                 cls = c1
                 
            else:
                 cls = c2
                 
            if(z[i,0] != cls):   #incorrect
               z[i,1] = z[i,1]*np.exp(alpha[0][q])
            else: #correct
               z[i,1] = z[i,1]*np.exp(-alpha[0][q])
        
        #normalize weights
        #print(sum(z[:]))
        z[:,1] = z[:,1]/sum(z[:,1])
    return alpha,hk
        
def oneAgainstOtherbatchadaboost(ftrain, ftest, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]
    features = ['Target']+features
    df = pd.read_csv(ftrain, names = features, sep=',')

    df_test = pd.read_csv(ftest,names = features, sep = ',')
    X = df[features[1:]]
    Y = df[features[0]]
    #margin
    #b = -0.1
    b = 5
    
    #number of classes
    c = 3
    # learning rate
    #eta = 0.01
    eta = 1
    
    #number of adaboost iterations
    kmax = 20

    #number of samples
    #s=1500
    s=50
    
    class1 = 2
    class2 = 3
    #Test data
    X_test = df_test[features[1:]]
    Y_test = df_test[features[0]]

    #first add 1 to feature to make augmented vector
    Ix  = np.ones((len(X_test), 1))
    
    #normalizing data
    X_norm = preprocessing.scale(X,axis=0,with_mean=True,with_std=True)
    X_test_norm = preprocessing.scale(X_test,axis=0,with_mean=True,with_std=True)
    
    #add weight, add 1, 
    X_test_norm = np.append(Ix,X_test_norm, axis = 1)
    #adaboost for class 1-2 classifier
    # call adaboost function
    alpha, hk = adaboost(X_norm, Y, s, class1, class2, eta, b, kmax)
    #print(hk)
    #g_ada will be computed by sum(alpha*hk(x))
    g_ada = 0;
    # Testing Adaboost for class1-2 classifier
    j = 0; h = 0; h1 = 0;
    #loop through each test sample
    for d in range(len(Y_test)):
        g_ada = 0;
        #test only class 1 and class 2 samples
        if(Y_test[d] == class1 or Y_test[d] == class2):
            j = j + 1;
            for l in range(kmax):
                if (np.dot(hk[l,:],X_test_norm[d,:]) > b):
                     g_ada = g_ada + alpha[0][l]
                else:
                     g_ada = g_ada - alpha[0][l]

            if g_ada > 0:
                cls = class1
            else:
                cls = class2
            
            if(Y_test[d] == cls):   #correct
                h = h+1;
              
            #%without adaboost
            if (np.dot(hk[1,:],X_test_norm[d,:]) > b):
                 cls = class1
            else:
                 cls = class2
             
            if(Y_test[d] == cls):   #correct
                h1 = h1+1

    p = h/j*100;
    p1 = h1/j*100;
    print('The performance of class ',class1,'-', class2,' classifier without AdaBoost on wine data set is =',p1)
    print('The performance of class ',class1,'-', class2,' classifier using AdaBoost on wine data set is =',p)
    
oneAgainstOtherbatchadaboost('./wine_train.txt','./wine_test.txt', 13)
#oneAgainstOtherbatchadaboost('./digit_train.txt','./digit_test.txt', 256)