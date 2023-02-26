import tensorflow as tf
import numpy as np
from tensorcircuit.templates.ensemble import bagging

data_amount = 100 # Amount of data to be used
linear_demension = 4 # linear demension of the data
epochs = 10
batch_size = 32
lr = 1e-3

x_train, y_train = (np.ones([data_amount, linear_demension]), np.ones([data_amount, 1]))

obj_bagging = bagging()

def model():

    DROP = 0.1

    activation = 'selu'
    inputs = tf.keras.Input(shape=(linear_demension,), name="digits")
    x0 = tf.keras.layers.Dense(1, 
              kernel_regularizer = tf.keras.regularizers.l2(9.613e-06),
              activation = activation,
             )(inputs)
    x0 = tf.keras.layers.Dropout(DROP)(x0)
    
    x = tf.keras.layers.Dense(1, 
                kernel_regularizer = tf.keras.regularizers.l2(1e-07),
                activation='sigmoid',
                )(x0)
    
    model = tf.keras.Model(inputs, x)
   
    return model

obj_bagging.append(model(), False)
obj_bagging.append(model(), False)
obj_bagging.append(model(), False)
obj_bagging.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr),
                metrics=[tf.keras.metrics.AUC(),'acc']
            )
obj_bagging.train(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size, verbose = 0)

v_weight = obj_bagging.predict(x_train, "weight")
v_most = obj_bagging.predict(x_train, "most")
v_average = obj_bagging.predict(x_train, "average")
validation_data = []
validation_data.append(obj_bagging.eval([y_train,v_weight],"acc"))
validation_data.append(obj_bagging.eval([y_train,v_weight],"auc"))