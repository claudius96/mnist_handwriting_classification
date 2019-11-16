import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#visualizing
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()

#import cnn datasets
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)
X_train =X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
#building our cnn model
def baseline_model():
    clf = Sequential()
    clf.add(Convolution2D(filters=32,kernel_size=(3),
                                       activation='relu',
                                       data_format='channels_last',
                                       input_shape=(28,28,1)))
    clf.add(Dropout(rate=0.2))
    clf.add(MaxPool2D(pool_size=(2,2)))
    clf.add(Flatten())
    #adding the full connection layer
    clf.add(Dense(units=128,activation='relu'))
    clf.add(Dense(output_dim=y_test.shape[1],activation='softmax'))
    
    #compiling the cnn
    clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return clf
clf = baseline_model()
clf.fit(X_train,y_train,batch_size=200,epochs=4,verbose=2,
        validation_data=(X_test, y_test))
scores =clf.evaluate(X_train, y_train,verbose=0)


