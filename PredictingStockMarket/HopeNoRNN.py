# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:07:33 2017

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:22:47 2017

@author: vishw
"""

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
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras

para= 1

train = read_csv('./ClassifiedRnn/trainREC.csv',sep=',')
test = read_csv('./ClassifiedRnn/testREC.csv',sep=',')

train = np.array(train)
test = np.array(test)[:len(test)-1]
trainX =  np.array(train[:,1:])
trainY =  np.array(train[:,0])
testY =  np.array(test[:,0])
testX =  np.array(test[:,1:])

feat = len(trainX[0])
classes = 3

trainSort = np.append(trainY, testY)
trainSort.sort()

breakPts = {}
for i in range(0, classes):
    breakPts[i] = trainSort[(len(trainSort)//classes)*i]

print(breakPts)
for i in range(len(trainY)):
    for j in reversed(list(breakPts)):
        if trainY[i] >= breakPts[j]:
           trainY[i] = j
           break
    for j in reversed(list(breakPts)):
        if testY[i] >= breakPts[j]:
            testY[i] = j
            break

trainY = keras.utils.to_categorical(trainY, num_classes=classes)
testY = keras.utils.to_categorical(testY, num_classes=classes)

lenX1 = len(test)
lenX2 = feat

trainX=np.array(trainX).reshape(lenX1,lenX2)
testX=np.array(testX).reshape(lenX1,lenX2)

trainY = np.array(trainY).reshape(lenX1,classes)
testY = np.array(testY).reshape(lenX1,classes)

model = Sequential()
n1 = 4000
n2 = 500
model.add(Dense(n1, input_shape=(lenX2,)))
model.add(Dense(n2))
model.add(Dense(n2))
#model.add(LSTM(n2, input_shape=(n2,1), return_sequences=True))
#model.add(LSTM(n2, input_shape=(n2,1), return_sequences=True))
model.add(Dense(classes))
stop = [EarlyStopping(monitor='categorical_accuracy', mode='max', min_delta=0.002, patience=5)]
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['categorical_accuracy'])
bsize = 5
model.fit(trainX, trainY, batch_size=bsize, epochs=100,shuffle=False, validation_data=(testX, testY), callbacks=stop)
predictions = model.predict(testX, batch_size=bsize)
peval = model.predict(trainX, batch_size=bsize)
evalsTe = model.evaluate(testX, testY)
evalsTr = model.evaluate(trainX, trainY)

predictionindex = []
for a in range(0,len(predictions)):
    predictionindex += [np.argmax(predictions[a])]

pevalindex = []
for a in range(0,len(peval)):
    pevalindex += [np.argmax(peval[a])]


testindex = []
for a in range(0,len(testY)):
    testindex += [np.argmax(testY[a])]

trainindex = []
for a in range(0,len(trainY)):
    trainindex += [np.argmax(trainY[a])]    
    
xte = np.linspace(1,len(testindex), num=len(testindex))
xtr = np.linspace(1,len(trainindex), num=len(trainindex))

ssize = 20
salpha = 0.5

testindex = np.array(testindex)
predictionindex = np.array(predictionindex)
trainindex = np.array(trainindex)
pevalindex = np.array(pevalindex)

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
