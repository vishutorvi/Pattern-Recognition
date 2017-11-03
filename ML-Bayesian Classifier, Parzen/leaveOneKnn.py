# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\Dell PC\.spyder2\.temp.py
"""
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from sklearn.model_selection import LeaveOneOut
import time

def analyzeKneighbors(ftrain,ftest, fsep, fcount, algorithm):
    features = ["w" + str(x) for x in range(1,fcount+1)]    
    features = ["Target"]+features
    df = pd.read_csv(ftrain,names = features,sep=fsep)
    X = df[features[1:]]
    Y = df[features[0]]

    X_norm = preprocessing.scale(X)
    
    #Finding proper k value from train set
    loo = LeaveOneOut()
    Misclassified={}
    Misclassifiedk=[]
    krange = list(range(1,11))
    for k in krange:
        break
        for cat in range(min(Y), max(Y) + 1):        
            prediction = 0
            i = 0
            for train, test in loo.split(X_norm):
                if Y[i] == cat:
                    clf = NearestNeighbors(n_neighbors=k,algorithm=algorithm,metric='manhattan')
                    clf.fit(X_norm[train], Y)
                    Xtest_final = np.array(X_norm[test])
                    distances,indices = clf.kneighbors(Xtest_final)
                    indices = indices[0]
                    Xtest_indices = []
                    for j in range(k):
                        indices_latest = indices[j]
                        Xtest_indices += [Y[indices_latest]]
                    max_indices =Counter(Xtest_indices).most_common(1)[0][0]
                    if Y[i] == max_indices:
                        prediction = prediction + 1
                i = i + 1
            Misclassified[cat]=[abs(len(Y)/max(Y) - prediction)]
        Misclassifiedk += [sum([Misclassified[x] for x in list(Misclassified)])]
        print 'Misclassifications for each cat is:',Misclassified,' for k value=',k                    
    
    #(1-np.array(Misclassifiedk)/len(Xtest_final))
#    plt.plot(krange, 1.0-(np.array(Misclassifiedk)*1.0)/len(Y))
#    plt.ylim(ymax = 1,ymin = 0)
#    plt.xlabel('k values')
#    plt.ylabel('Accuracy')
#    plt.title('Leave one out accuracy comparison for handwritten data set-KNN')
#    #k = krange[np.argmin(Misclassifiedk)]
    k = 3    
    print 'k min=',k
    #After getting k value testing on test data       
    test = pd.read_csv(ftest,names =features,sep=fsep)
    Xtest = test[features[1:]]
    Ytest = test[features[0]]
    print('Len of Y=0',len(Ytest[Ytest==0]))
    print('Len of Y=1',len(Ytest[Ytest==1]))
    print('Len of Y=2',len(Ytest[Ytest==2]))
    Xtest_norm= preprocessing.scale(Xtest)
    clf = NearestNeighbors(n_neighbors=k,algorithm=algorithm,metric='manhattan')
    clf.fit(X_norm, Y)
    start_time = time.time()
    distances,indices = clf.kneighbors(Xtest_norm)#Xtest_norm)
    predictions = [[0 for x in range(max(Ytest) + 1)] for x in Ytest]
    for i in range(k):
        index = [x[i] for x in indices]
        prediction = np.array(Y[index])
        for j in range(len(predictions)):
            predictions[j][prediction[j]] += 1
    prediction = [predictions[i].index(max(predictions[i])) for i in range(len(Ytest))]

    print len(Ytest)        
    print 'Mis-Prediction:',np.sum(Ytest!=prediction),'for test set'
    print 'predicted as=',prediction
    print 'Accuracy:',((len(Ytest)-np.sum(Ytest!=prediction))/float(len(Ytest)))*100
    print 'Time taken for',ftest,'using algorithm=', algorithm,'is=',(time.time()-start_time)*1000,'Milliseconds'

#with K-d tree
#analyzeKneighbors("irisdataset.csv", "iristestdata.csv", ',', 4,10,'kd_tree')
#analyzeKneighbors("wine_uci_train.txt","wine_uci_test.txt",',',13, 9,'kd_tree')
analyzeKneighbors("sentimentdata.csv","sentimenttest.csv",',',500,'kd_tree')
#analyzeKneighbors("zip_train_small.txt","ownWritten.txt",',',256, 92,'kd_tree')
#with auto algorithm
#analyzeKneighbors("irisdataset.csv", "iristestdata.csv", ',', 4,10,'brute')
#analyzeKneighbors("wine_uci_train.txt","wine_uci_test.txt",',',13, 9,'brute')
#analyzeKneighbors("zip_train_small.txt","zip_test.txt",',',256, 92,'brute')