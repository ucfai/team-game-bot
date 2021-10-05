import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Dense(3, input_dim=9, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

#learning_rate = 0.001
#momentum = 0.8

#sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
model.compile(loss='mean_squared_error')
model.summary()
