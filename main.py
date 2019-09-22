#!/usr/bin/env python3
from __future__ import print_function
import IPython
import sys
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

reshapor = Reshape((1, 78))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')
