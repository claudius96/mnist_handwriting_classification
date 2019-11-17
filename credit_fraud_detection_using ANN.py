
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load the Dataset
ds = pd.read_csv('creditcard.csv')
X = ds.iloc[:, 0:-1].values
y = ds.iloc[:, -1].values

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Feature scaling of data variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#importing the libraries for the ann algorithm
import keras
from keras.layers import Dense
from keras.models import Sequential

clf = Sequential()
clf.add(Dense(units=15,activation='relu',kernel_initializer='uniform',
              input_dim=30))
clf.add(Dense(units=15,activation='relu'))
clf.add(Dense(units=1,activation='sigmoid'))

#compiling the ann
clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

clf.fit(X_train, y_train,epochs=20,batch_size=10)

y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

clf.evaluate(X_test, y_test)















