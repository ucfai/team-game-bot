import mnk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adadelta

learning_rate = 1.0
rho = 0.7
epsilon = 1e-07
sgd = Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon)

modelXO = Sequential()
modelXO.add(Dense(27, input_shape=(1,9), kernel_initializer='normal', activation='tanh'))
modelXO.add(Dense(18, kernel_initializer='normal', activation='tanh'))
modelXO.add(Dense(1, kernel_initializer='normal', activation='tanh'))

modelXO.compile(loss='mean_squared_error', optimizer=sgd)

