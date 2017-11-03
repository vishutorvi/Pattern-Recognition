# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:22:42 2017

@author: Dell PC
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:03:41 2017

@author: Dell PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.neighbors import KernelDensity

def analysis(ftrain,ftest, fsep, fcount, band):
    features = ["w" + str(x) for x in range(1,fcount+1)]    
    features = ["Target"]+features
    df = pd.read_csv(ftrain,names = features,sep=fsep)
    X = df[features[1:]]
    Y = df[features[0]]
    h_range = np.linspace(0.1,1,num=10)
    X_norm = preprocessing.scale(X)
    leave_one_out = []
    for h in h_range:
        Ytest = Y
        prediction = []
        for j in range(len(X_norm)):
            Xtest_norm_in = [X_norm[j]]
            Y_in = np.array(Y)
            Y_in = np.delete(Y_in, j)
            X_norm_in = X_norm
            X_norm_in = np.delete(X_norm_in, j, 0)
            score = []    
            for cat in range(min(Y_in), max(Y_in) + 1):
                clf = KernelDensity(bandwidth=h)
                X_norm_clf_in = []
                for i in range(len(Y_in)):
                    if Y[i] == cat:
                        X_norm_clf_in += [X_norm_in[i]]
                clf.fit(X_norm_clf_in)
                score += [clf.score_samples(Xtest_norm_in)]
            score = np.array(score)
            for i in range(len(Xtest_norm_in)):
                prediction += [np.argmax(score[:,i]) + min(Y_in)]
        leave_one_out += [np.sum(prediction!=Ytest)]
    hedge_idx = np.argmin(leave_one_out)
    
#    plt.plot(h_range, 1-((np.array(leave_one_out))*1.0)/len(Y))
#    plt.xlabel('Window width values')
#    plt.ylabel('Accuracy')
#    plt.title('Leave one out Accuracy Estimation for wine dataset-Parzen Window')
#    print 'Results from leave one out comparison:\n'
#    print 'Misclassification for different bandwidth ranges:\n'
#    print '----------------------------------------------------------------------\n'
#    print '        bandwidth range:', h_range
#    print 'Misclassification count:',leave_one_out
#    hedge_op =  h_range[hedge_idx]
#    print 'we choose bandwidth=',hedge_op,',this gives less error while performing leave one out\n'
#    print '----------------------------------------------------------------------\n'
    #X_df will have all the values
    test = pd.read_csv(ftest,names =features,sep=fsep)
    Xtest = test[features[1:]]
    Ytest = test[features[0]]
    Xtest_norm = preprocessing.scale(Xtest)  
    Xtest_norm = np.array(Xtest_norm)
    score = []    
    for cat in range(min(Y), max(Y) + 1):
        clf = KernelDensity(bandwidth=band)
        X_norm_clf = []
        for i in range(len(Y)):
            if Y[i] == cat:
                X_norm_clf += [X_norm[i]]
        clf.fit(X_norm_clf)
        score += [clf.score_samples(Xtest_norm)]
    score = np.array(score)
    prediction = []
    for i in range(len(Xtest_norm)):
        prediction += [np.argmax(score[:,i]) + min(Y)]
    print 'Misclassified on test data:',np.sum(prediction!=Ytest),'for',ftest,'\n'
    print '----------------------------------------------------------------------\n'
    print 'Predicted as=',prediction
    print 'Accuracy on the test data:',((len(Ytest)-np.sum(Ytest!=prediction))/float(len(Ytest)))*100
    print '----------------------------------------------------------------------\n'

#analysis("iristestdata.csv","irisdataset.csv", ',', 4, 0.3)
#analysis("wine_uci_train.txt","wine_uci_test.txt",',',13, 0.7)
#analysis("zip_train_small.txt","ownWritten.txt",',',256, 0.9)
analysis("sentimentdata.csv","sentimenttest.csv",',',500, 0.2)