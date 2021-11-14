import mnk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD

learning_rate = 0.01
momentum = 0.0
sgd = SGD(learning_rate=learning_rate, momentum=momentum)

modelXO = Sequential()
modelXO.add(Conv2D(12, 5, padding="valid", input_shape=(15,15,2)))
modelXO.add(Conv2D(12, 5, padding="valid", input_shape=(15,15,2)))
modelXO.add(MaxPooling2D((2,2)))
modelXO.add(Flatten())
modelXO.add(Dense(27, kernel_initializer='normal', activation='tanh'))
modelXO.add(Dense(1, kernel_initializer='normal', activation='tanh'))

modelXO.compile(loss='mean_squared_error', optimizer=sgd)

