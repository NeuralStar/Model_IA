#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers.convolutional import Conv1D
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.layers import TimeDistributed
from keras.models import Model
from keras import regularizers
from keras.layers  import  LSTM
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector
from keras.models import Model
import tensorflow as tf
import math, sys, time
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
# Multiple Inputs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
#from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
import math, sys, time
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, TimeDistributed, LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from pandas._testing import assert_frame_equal
from pandas.testing import assert_index_equal
from Gate_activation import GatedActivationUnit
from sklearn.datasets import make_classification
from sklearn.utils import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

class WaveNet_EEGModel:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = self.create_model()

    def create_model(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        z = keras.layers.Conv1D(filters=120, kernel_size=1, strides=1, padding="valid")(inputs)
        skip_to_last = []
        for dilation_rate in [2**i for i in range(10)] * 3:
            z, skip = self.wavenet_residual_block(z, 120, dilation_rate)
            skip_to_last.append(skip)
        z = keras.activations.relu(keras.layers.Add()(skip_to_last))
        z = keras.layers.Conv1D(120, kernel_size=1, activation="relu")(z)
        z = keras.layers.GlobalAveragePooling1D()(z)
        outputs = keras.layers.Dense(self.n_classes, activation="softmax")(z)
        model = keras.models.Model(inputs=[inputs], outputs=[outputs])
        return model

    def wavenet_residual_block(self, inputs, n_filters, dilation_rate):
        z = keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                                dilation_rate=dilation_rate)(inputs)
        z = GatedActivationUnit()(z)
        #z = keras.layers.Bidirectional(LSTM(n_filters, activation='relu', return_sequences=True))(z)
        z = GatedActivationUnit()(z)
        #z = keras.layers.Bidirectional(GRU(n_filters, activation='relu', return_sequences=True))(z)
        z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
        return keras.layers.Add()([z, inputs]), z

    def fit(self, X_train, y_train, X_val, y_val):
        checkpoint_cb = ModelCheckpoint('Al_Hajj',save_best_only=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Calculer les poids pour chaque classe
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight = "balanced", classes= classes, y= y_train)
        class_weights = {i: w for i, w in enumerate(class_weights)}
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=100, batch_size=100, validation_data=(X_val, y_val), shuffle=True, 
                       callbacks=[checkpoint_cb,tensorboard_callback], class_weight=class_weights,)






