import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import LearningRateScheduler, History, EarlyStopping
from tensorflow.keras import backend as K
# from: https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor
# https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
from keras.wrappers.scikit_learn import KerasRegressor

from aem.data import load_data
from aem.config import Config

config_file = '/home/sudipta/repos/aem-ml/configs/tfregression.yaml'

conf = Config(config_file)

X, y, w = load_data(conf)

epochs = 100
learning_rate = 0.1  # initial learning rate
decay_rate = 0.1
momentum = 0.8

normalizer = preprocessing.Normalization()

def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

# learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)
early_stopping = EarlyStopping(monitor='loss', min_delta=1.0e-6, verbose=1, patience=10)
callbacks_list = [loss_history, lr_rate, early_stopping]

# TODO: Tensorflow or a DNN regression class


class KerasRegressorWrapper(KerasRegressor):

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, **kwargs)


def build_and_compile_model(self, X, y, norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation='linear')
    ])
    return model

    # model.compile(loss='mean_absolute_error',
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #               metrics=['mean_absolute_error', 'mean_squared_error', r2_score]
    #               )


tfreg = KerasRegressorWrapper(build_fn=build_and_compile_model)

tfreg.fit(X_train, y)
y_pred = tfreg.predict(X_test)


