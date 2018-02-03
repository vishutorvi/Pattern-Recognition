# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:57:57 2017

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:31:25 2017

@author: vishw
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

para= 6

train = read_csv('./ClassifiedRnn/trainREC.csv',sep=',')
test = read_csv('./ClassifiedRnn/testREC.csv',sep=',')

train = np.array(train)
test = np.array(test)[:len(test)-1]
trainX =  np.array(train[:,1:])
trainY =  np.array(train[:,0])
testY =  np.array(test[:,0])
testX =  np.array(test[:,1:])

feat = len(trainX[0])
classes = 2

trainSort = np.append(trainY, testY)
trainSort.sort()

breakPts = {}
for i in range(0, classes):
    breakPts[i] = trainSort[(len(trainSort)//classes)*i]

lenX1 = len(test)
lenX2 = feat

trainXMod = []
trainYMod = []
for i in range(para-1,len(trainX)):
    trainXModTmp = []
    trainYModTmp = []
    
    for j in range(para-1,-1,-1):
        trainXModTmp += [trainX[i-j]]
        
        cls = 0
        for bp in breakPts:
            if(trainY[i-j] >= breakPts[bp]):
                cls = bp
                
        clss = []
        for bp in breakPts: 
            if cls == bp:
                clss += [1]
            else:
                clss += [0]
        if j == 0:
            for k in range(para-1,-1,-1):
                trainYModTmp += [clss]
    
    trainXMod += [trainXModTmp]
    trainYMod += [trainYModTmp]
    
testXMod = []
testYMod = []
for i in range(para-1,len(testX)):
    testXModTmp = []
    testYModTmp = []
    
    for j in range(para-1,-1,-1):
        testXModTmp += [testX[i-j]]
        
        cls = 0
        for bp in breakPts:
            if(testY[i-j] >= breakPts[bp]):
                cls = bp
                
        clss = []
        for bp in breakPts: 
            if cls == bp:
                clss += [1]
            else:
                clss += [0]
        
        if j == 0:
            for k in range(para-1,-1,-1):
                testYModTmp += [clss]
    
    testXMod += [testXModTmp]
    testYMod += [testYModTmp]

trainX=np.array(trainXMod)
testX=np.array(testXMod)

trainY = np.array(trainYMod)
testY = np.array(testYMod)

lenX1 = para
lenX2 = feat

model = Sequential()
n1 = 1200
n2 = 400
model.add(LSTM(n1, input_shape=(lenX1, lenX2), return_sequences=True))
model.add(LSTM(n2, input_shape=(n1, lenX1), return_sequences=True))
model.add(LSTM(n2, input_shape=(n2, lenX1), return_sequences=True))
model.add(LSTM(n2, input_shape=(n2, lenX1), return_sequences=True))
model.add(LSTM(n2, input_shape=(n2, lenX1), return_sequences=True))
model.add(Dense(len(breakPts)))
stop = [EarlyStopping(monitor='categorical_accuracy', mode='max', min_delta=0.02, patience=100)]
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['categorical_accuracy'])
trainSize = 1
bsize = 5
maxEpochs = 100
bestAcc = 0
for i in range(maxEpochs//trainSize):
    print(i * trainSize)
    model.fit(trainX, trainY, batch_size=bsize, epochs=trainSize,shuffle=False, validation_data=(testX, testY))
    evalsTe = model.evaluate(testX, testY)[1]
    evalsTr = model.evaluate(trainX, trainY)[1]
    if min(evalsTe,evalsTe) > bestAcc:
        bestAcc = min(evalsTe,evalsTe)
        model.save_weights('model_weights.h5')
model.load_weights('model_weights.h5')

predictions = model.predict(testX, batch_size=bsize)
peval = model.predict(trainX, batch_size=bsize)
evalsTe = model.evaluate(testX, testY)
evalsTr = model.evaluate(trainX, trainY)

predictionindex = []
for a in range(0,len(predictions)):
    predictionindex += [np.argmax(predictions[a][-1])]

pevalindex = []
for a in range(0,len(peval)):
    pevalindex += [np.argmax(peval[a][-1])]


testindex = []
for a in range(0,len(testY)):
    testindex += [np.argmax(testY[a,0])]

trainindex = []
for a in range(0,len(trainY)):
    trainindex += [np.argmax(trainY[a,0])]    
    
testindex = np.array(testindex)
predictionindex = np.array(predictionindex)
trainindex = np.array(trainindex)
pevalindex = np.array(pevalindex)

mid = classes//2
redTest = []
redPred = []
redTrain = []
redPeval = []

for arrI, arrO in [[testindex,redTest], [predictionindex, redPred], [trainindex, redTrain], [pevalindex,redPeval]]:
    for i in range(len(arrI)):
        if arrI[i] >= mid:
            arrO += [1]
        else:
            arrO += [0]

#testindex = np.array(redTest)
#predictionindex = np.array(redPred)
#trainindex = np.array(redTrain)
#pevalindex = np.array(redPeval)

xte = np.linspace(1,len(testindex), num=len(testindex))
xtr = np.linspace(1,len(trainindex), num=len(trainindex))

ssize = 20
salpha = 0.5


plt.figure(1)
plt.scatter(xte,testindex, c='r',alpha=salpha, s=ssize)
plt.scatter(xte,predictionindex, c='b',alpha=salpha, s=ssize)
plt.figure(2)
plt.scatter(xtr,trainindex, c='r',alpha=salpha, s=ssize)
plt.scatter(xtr,pevalindex, c='b',alpha=salpha, s=ssize)
plt.show()

accTe = np.sum(testindex==predictionindex)
accTr = np.sum(trainindex==pevalindex)

print('Train Acc:', accTr)
print('Train Acc%:', accTr/len(trainindex) * 100)
print('Test Acc:', accTe)
print('Train Acc%:', accTe/len(testindex) * 100)
