# imports
from keras.layers import Input, Dense,Conv1D, BatchNormalization, AveragePooling1D , UpSampling1D, Activation, Flatten, Reshape
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras import regularizers
import numpy as np

traces= 56          # number of sensors 
trace_length = 176  # number of time stamps per sensor

# definition of the encoder part
def encoder(input_data):
    tcn1 = Conv1D(64, kernel_size=3, padding='causal') (input_data)
    activ1= Activation('relu')(tcn1)
    pool1 = AveragePooling1D(pool_size=2)(activ1)
    tcn2 = Conv1D(64, kernel_size=3, padding='causal')(pool1)
    activ2= Activation('relu')(tcn2)
    pool2 = AveragePooling1D(pool_size=2)(activ2)
    tcn3 = Conv1D(64, kernel_size=3, padding='causal')(pool2)
    activ3= Activation('relu')(tcn3)
    pool3 = AveragePooling1D(pool_size=2)(activ3)
    flatten3= Flatten()(pool3)

    return flatten3

# definition of the decoder part
def decoder(merge,pool21=22,pool22=7):  

    dense2 = Dense(pool21*pool22, activation='relu')(merge)
    reshaped = Reshape((pool21,pool22,))(dense2)
    tcn3 =Conv1D(64, kernel_size=3, padding='causal') (reshaped)
    activ3= Activation('relu')(tcn3)
    up2 = UpSampling1D((2))(activ3) 
    tcn4 = Conv1D(64, kernel_size=3, padding='causal') (up2)
    activ4= Activation('relu')(tcn4)
    up3 = UpSampling1D((2))(activ4) 
    tcn5 = Conv1D(64, kernel_size=3, padding='causal') (up3)
    activ5= Activation('relu')(tcn5)
    up4 = UpSampling1D((2))(activ5) 

    decoded = Conv1D(traces, kernel_size=3, padding='causal') (up4) 
    return decoded


# input
input_data = Input(shape=(trace_length,traces))

# the encoder model
encode= encoder(input_data)
encoder = Model(input_data, encode)

# autoencoder model
autoencoder = Model(input_data, decoder(encode))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)  
autoencoder.summary()
autoencoder.compile(optimizer=optimizer, loss= 'mean_squared_error', metrics=['mse'])

# weight transfer
encoder.set_weights(autoencoder.get_weights()[:11])