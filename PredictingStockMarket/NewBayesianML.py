# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:07:38 2017

@author: Dell PC
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from sklearn.pipeline import Pipeline

class bayesClassifier(object):
    def fit(self, X,Y):
        #N, D=X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y==c]
            self.gaussians[c]={
                'mean':current_x.mean(axis=0),
                'var':np.var(current_x)
            }
            self.priors[c] = float(len(Y[Y==c])) / len(Y)
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
    def predict(self,X):
        N, D = X.shape
        B = 0.1
        K = len(self.gaussians)
        P = np.zeros((N, K+1))
        #P_log = np.zeros((N, K+1))
        for c,g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            mean = np.array(mean)
            var = (1-B)*np.array(var) + B*np.eye(len(var))
            #print 'Category =',c,'\n'
            #print '\n---------------------------------------------------\n'
            #print 'mean =',mean,'for category=',c,'\n'
            #print 'variance=', var,'for category=',c
            #print '\n---------------------------------------------------\n'
            P[:,c] = mvn.pdf(X, mean=mean, cov=var)*self.priors[c]
            #mvn.pdf()
        return np.argmax(P, axis = 1)

def analysis(ftrain,ftest, fsep, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]    
    features = ["Target"]+features
    df = pd.read_csv(ftrain,names = features,sep=fsep)
    X = df[features[1:]]
    Y = df[features[0]]
    X_norm = X
    test = pd.read_csv(ftest,names=features,sep=fsep)
    Xtest = test[features[1:]]
    Ytest = test[features[0]]
    Xtest_norm = Xtest
    clf = Pipeline([
    ('classification', bayesClassifier())
    ])
    clf.fit(X_norm, Y)
    prediction = clf.predict(Xtest_norm)
    print('Len of Y=0',len(Ytest[Ytest==0]))
    print('Len of Y=1',len(Ytest[Ytest==1]))
    #print('Len of Y=2',len(Ytest[Ytest==2]))
    #print '\n---------------------------------------------------\n'   
    print(prediction)
    print('Misclassified:',np.sum(prediction!=Ytest))
    print('Accuracy:',((len(Ytest)-np.sum(Ytest!=prediction))/float(len(Ytest)))*100)
    #print '\n---------------------------------------------------\n'

#analysis("irisdataset.csv", "iristestdata.csv", ',', 4)
analysis("sentimentdata.csv","sentimenttest.csv",',',2000)
#analysis("zip_train_small.txt","ownWritten.txt",',',256)