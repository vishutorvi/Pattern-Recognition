# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:07:38 2017

@author: Dell PC
"""
import numpy as np
import pandas as pd
#import seaborn as sns
#import sklearn as sk
import operator
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def analysis(ftrain,ftest, fsep, fcount):
    features = ["w" + str(x) for x in range(1,fcount+1)]    
    features = ["Target"]+features
    df = pd.read_csv(ftrain,names = features,sep=fsep)
    X = df[features[1:]]
    Y = df[features[0]]
    print(X)
    #print 'Covariance Matrix=',np.cov(X)
    # feature extraction Using Recursive Feature Elimination Technique
    #model = LogisticRegression()
    print(min(len(Y)-1,fcount))
    #rfe = RFE(model, min(len(Y)-1,fcount))
    #fit = rfe.fit(X, Y)
    #nprank = np.array(fit.ranking_)
    X_norm = preprocessing.scale(X)
    #columnInd = np.argwhere(nprank==1)
    #columnInd = [x for [x] in columnInd]
    #X_df = pd.DataFrame(X_norm[:,columnInd])
    #X_norm = np.array(X)
    X_norm = X_norm - np.min(X_norm)
    print(np.min(X_norm))
    clf = MultinomialNB()
    clf.fit(X_norm, Y)
    #print(clf.)
    test = pd.read_csv(ftest,names=features,sep=fsep)
    Xtest = test[features[1:]]
    Ytest = test[features[0]]
    Xtest_norm= preprocessing.scale(Xtest)
    #print 'theta=',clf.__getattribute__('theta_')
    #print('sigma=',clf.__getattribute__('sigma_'))
    #print('Covariance=',np.cov(X_norm))
    Xtest_df = pd.DataFrame(Xtest_norm)
    Xtest_final = np.array(Xtest_df)
    #print Xtest_final
    
    prediction = clf.predict(Xtest_final)
#    misMatch = {}
#    for i in range(len(prediction)):
#        if prediction[i] != Ytest[i]:
#            if str(prediction[i]) + '=' + str(Ytest[i]) in misMatch:
#                misMatch[str(prediction[i]) + '=' + str(Ytest[i])] += 1
#            else:
#                misMatch[str(prediction[i]) + '=' + str(Ytest[i])] = 1
#    print sorted(misMatch.items(), key=operator.itemgetter(1), reverse=True)
    predictionAdj = [1 for x in range(len(prediction))]   
    print(list(Ytest))    
    print(list(prediction))
    #print(predictionAdj)
    #print(list(np.abs(predictionAdj-Ytest)))
    print('Prior=',np.sum(Ytest))
    print('PriorAdj=',(np.sum(np.abs(predictionAdj-Ytest)) / len(Ytest)))
    print('Miscalssified Adjusted', (np.sum(np.abs(prediction-Ytest)) / len(Ytest)))
    print('Misclassified:',np.sum(prediction!=Ytest))
    print('Accuracy:',(float(len(Ytest)-np.sum(prediction!=Ytest))/len(Ytest))*100)

#analysis("irisdataset.csv", "iristestdata.csv", ',', 4)
#analysis("wine_uci_train.txt","wine_uci_test.txt",',',13)
analysis("sentimentdata.csv","sentimenttest.csv",',',2000)
#analysis("zip_train_small.txt","zip_test.txt",',',256)