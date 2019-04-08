import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, optimizers
from keras import models
from os.path import join
from settings import *


train_set = join(DATA_SPLIT_ROOT, 'train.csv')
test_set = join(DATA_SPLIT_ROOT, 'test.csv')

train = pd.read_csv(train_set, encoding='latin1', low_memory=True)
test = pd.read_csv(test_set, encoding='latin1', low_memory=True)

train_x = train.drop(['success'], axis=1)
train_y = train.success

test_x = test.drop(['success'], axis=1)
test_y = test.success

print(train_x.columns)

# model = models.Sequential()
# # Input - Layer
# model.add(layers.Dense(1000, activation="relu", input_shape=(225,)))
# # Hidden - Layers
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# model.add(layers.Dense(200, activation="relu"))
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# model.add(layers.Dense(200, activation="relu"))
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# model.add(layers.Dense(200, activation="relu"))
# # Output- Layer
# model.add(layers.Dense(1, activation=tf.nn.sigmoid))
# model.summary()
# adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# # compiling the model
# model.compile(
#     optimizer=adadelta,
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )
# results = model.fit(
#     train_x, train_y,
#     epochs=100,
#     batch_size=500,
#     validation_data=(test_x, test_y)
# )
# print("Test-Accuracy:", np.mean(results.history["val_acc"]))