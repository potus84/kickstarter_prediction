#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
os.chdir('/home/potusvn/Projects/kickstarter_prediction')



import numpy as np
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe

from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils
from settings import *
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform



def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # train_set = os.path.join(DATA_SPLIT_ROOT, 'train.csv')
    # test_set = os.path.join(DATA_SPLIT_ROOT, 'test.csv')
    #
    # train = pd.read_csv(train_set, encoding='latin1', low_memory=True)
    # test = pd.read_csv(test_set, encoding='latin1', low_memory=True)
    #
    # x_train = train.drop(['success','country_nan','currency_nan'], axis=1)
    # y_train = train.success
    #
    # x_test = test.drop(['success','country_nan','currency_nan'], axis=1)
    # y_test = test.success
    #
    # return x_train, y_train, x_test, y_test
    train = pd.read_csv('data/train_quantile_transform.csv', encoding='latin1', low_memory=True)
    test = pd.read_csv('data/test_quantile_transform.csv', encoding='latin1', low_memory=True)
    val = pd.read_csv('data/val_quantile_transform.csv', encoding='latin1', low_memory=True)

    train_x = train.drop(['success'], axis=1)
    train_y = train.success

    val_x = val.drop(['success'], axis=1)
    val_y = val.success

    test_x = test.drop(['success'], axis=1)
    test_y = test.success

    return train_x, train_y, val_x, val_y, test_x, test_y


# In[18]:


def create_model(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(
        {{choice([10, 20, 25])}},
        activation="relu", input_shape=(221,),
    ))
    model.add(Dropout({{uniform(0, 1)}}))

    model = Sequential()
    model.add(Dense(
        {{choice([10, 20, 25])}},
        activation="relu", input_shape=(221,),
    ))
    model.add(Dropout({{uniform(0, 1)}}))

    model = Sequential()
    model.add(Dense(
        {{choice([10, 20, 25])}},
        activation="relu", input_shape=(221,),
    ))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    result = model.fit(train_x, train_y,
              batch_size={{choice([512, 1024])}},
              epochs={{choice([100, 200, 300])}},
              validation_data=(val_x, val_y)
            )
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


# In[15]:


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          eval_space=True,
                                          max_evals=10,
                                          trials=Trials(),
                                          notebook_name='src/models/Hyperas')
    X_train, Y_train, _, _, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)






