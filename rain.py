# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:32:00 2018

@author: Martin
"""

import numpy as np 
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
import seaborn as sns
import keras

dataset=pd.read_csv('rain.csv')
X=dataset.iloc[:,2:4].values
y=dataset.iloc[:,-1].values
prcp=dataset.iloc[:,1:2].values

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
prcp_train, prcp_test=train_test_split(prcp, test_size=0.25,random_state=0)

from sklearn import preprocessing
sc=preprocessing.MinMaxScaler()
X_train_sc=sc.fit_transform(X_train)
X_train=np.concatenate((prcp_train,X_train_sc),axis=1)
X_test_sc=sc.fit_transform(X_test)
X_test=np.concatenate((prcp_test,X_test_sc),axis=1)

classifier=keras.Sequential()

classifier.add(Dense(units=4,kernel_initializer='uniform',activation='relu',input_dim=3))
classifier.add(Dropout(rate=0.10))
classifier.add(Dense(units=4,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.10))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=30,epochs=16)

y_predicted=classifier.predict(X_test)
y_predicted=(y_predicted>0.5)

score = classifier.evaluate(X_test, y_test, verbose=0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)


