import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

trainE = 100
trainN = 4
look_back = 249
train = read_csv('./ClassifiedRnn/trainREC.csv',sep=',')
test = read_csv('./ClassifiedRnn/testREC.csv',sep=',')

train = np.array(train)
test = np.array(test)[:len(test)-1]
trainX =  np.array(train[:,1:]).flatten()
trainY =  np.array(train[:,0]).flatten()
testY =  np.array(test[:,0]).flatten()
testX =  np.array(test[:,1:]).flatten()

lenX1 = 249
lenX2 = 1000

trainX = trainX.reshape(1, lenX2, lenX1)
trainY = trainY.reshape(1, lenX1)

testX = testX.reshape(1, lenX2, lenX1)
testY= testY.reshape(1, lenX1)

model = Sequential()
model.add(LSTM(100, input_shape=(lenX2, lenX1), return_sequences=True, stateful=False))
#model.add(LSTM(100, input_shape=(100, lenX1), return_sequences=True))
model.add(LSTM(100, input_shape=(100, lenX1)))
model.add(Dense(lenX1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, batch_size=1, epochs=20, validation_data=(testX, testY),shuffle=False)
predictions = model.predict(testX, batch_size=1)
peval = model.predict(trainX, batch_size=1)
evals = model.evaluate(testX, testY)

x = np.linspace(1,249, num=249)

#plt.plot(x,trainY[0], c='r')
#plt.plot(x,peval[0], c='b')
plt.show()
plt.plot(x,testY[0], c='r')
plt.plot(x,predictions[0], c='b')
plt.show()