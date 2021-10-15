import mnk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

learning_rate = 0.005
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)

modelXO = Sequential()
modelXO.add(Dense(3, input_dim=9, kernel_initializer='normal', activation='tanh'))
modelXO.add(Dense(1, kernel_initializer='normal', activation='tanh'))

modelXO.compile(loss='mean_squared_error', optimizer = sgd)

