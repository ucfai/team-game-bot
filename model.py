import mnk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

learning_rate = 0.005
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)

modelX = Sequential()
modelX.add(Dense(3, input_dim=9, kernel_initializer='normal', activation='sigmoid'))
modelX.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

modelX.compile(loss='mean_squared_error', optimizer = sgd)

######################################################################

modelO = Sequential()
modelO.add(Dense(3, input_dim=9, kernel_initializer='normal', activation='sigmoid'))
modelO.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

modelO.compile(loss='mean_squared_error', optimizer = sgd)
