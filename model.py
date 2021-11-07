import mnk
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adadelta

learning_rate = 1.0
rho = 0.995
epsilon = 1e-07
sgd = Adadelta(lr=learning_rate, rho=rho, epsilon=epsilon)

modelXO = Sequential()
modelXO.add(Conv2D(12, 3, padding="same", input_shape=(3, 3, 1), activation='tanh', kernel_initializer="he_normal"))
modelXO.add(Dropout(0.1))
modelXO.add(Conv2D(9, 2, padding="valid", input_shape=(3, 3, 1), activation='tanh', kernel_initializer="he_normal"))
modelXO.add(Dropout(0.1))
modelXO.add(Flatten())
modelXO.add(Dense(18, kernel_initializer='normal', activation='tanh'))
modelXO.add(Dense(1, kernel_initializer='normal', activation='tanh'))

modelXO.compile(loss='mean_squared_error', optimizer=sgd)

