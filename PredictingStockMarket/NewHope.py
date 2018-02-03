# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:25:56 2017

@author: vishw
"""

import pyrenn as prn
import numpy as np
import pandas as pd

train = pd.read_csv('C:/TermProject/ClassifiedRnn/trainREC.csv',sep=',',header=None)
test = pd.read_csv('C:/TermProject/ClassifiedRnn/testREC.csv',sep=',',header=None)

train = np.array(train)
test = np.array(test)
net = prn.CreateNN([1000,20,1],dIn=[0],dIntern=[1],dOut = [1,2])

net = prn.train_LM(train[:,1:].T,train[:,0],net,verbose=True,k_max=30,E_stop=1e-3)

y = prn.NNOut(train[:,1:].T,net)
ytest = prn.NNOut(test[:,1:].T,net)

yTestPrd = np.array(ytest)
yTrainPrd = np.array(y)
yTestCor = test[:,0]
yTrainCor = train[:,0]
difTest = yTestPrd - yTestCor
difTrain = yTrainPrd - yTrainCor

accTest = np.mean(np.abs(difTest))
accTrain = np.mean(np.abs(difTrain))

print(accTrain)
print(accTest)

acc = 0.0
for i in range(len(yTrainCor)):
    cor = yTrainCor[i]
    pred = yTestPrd[i]
    
    if cor >= 3 and pred >= 3:
        acc += 1
    elif cor <= -3 and pred <= -3:
        acc += 1
    elif cor < 3 and pred < 3 and cor > -3 and pred > -3:
        acc += 1
print(acc/len(yTrainCor))