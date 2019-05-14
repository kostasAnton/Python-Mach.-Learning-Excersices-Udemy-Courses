#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:41:34 2019

@author: kostas
"""
#https://www.udemy.com/deeplearning/learn/v4/questions/3554002
#multiple indicators
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the training set on variable
dataset_training=pd.read_csv('/home/kostas/Desktop/Udemy/excersice 3/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
training_Set=dataset_training.iloc[:,1:2].values

#feuature scaling because we want values between 0 and 1
#we dont want much fluctuation
#https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range = (0,2))
training_Set_scaled = sc.fit_transform(training_Set)

#each time we are going to use the 60 previous stocks to predict the next one
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_Set_scaled[i-60:i,0])
    y_train.append(training_Set_scaled[i,0])
X_train, y_train=np.array(X_train), np.array(y_train)

#reshape the data add a new dimension
#it is the unit which concers the number of predictors we are going to use in
# order to rpedict the outcome(predicts concerns the stock price.The time 't+1'
#unit is actually the number of predictors,we can use to predict what we want.
#These predictors are indicators ..now we have one stock price as indicator(feature) 
#we will add a new dimension as extra indicator so we can predict even better
#Input shape-newshape  the second arguement of reshape
#3D tensor with shape (batch_size, timesteps, input_dim) 
#->the three dimension of X_train after the reshape .
#in the parenthesis ..we put the number of lines then the number of timesteps 
#and finally the number of predictors
#This is what our neural network will expect
X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#we are predictin a continuous output
regressor=Sequential()
#memory units or neurons i want in the 1st layer . bIg number for units because 
#we need to capture the trends between the stock price
#return sequence true because we will add another layer
#the number of stock prices (1258) will be taken account automatically by keras
#basically our input shape is not a 2s dimmension but a 3d however we specify 
#the 2dimensions
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
#20% of neuros during forward propagation and backpropagation is going to be 
#ingored
#DROPOUT DURING EACH ITERATION OF THE TRAINING
regressor.add(Dropout(0.20))

#second lstm layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.20))


#thrid lstm layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.20))

#fourth lstm layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.20))

#add the poutput layer
regressor.add(Dense(units=1))
#compiling the RNN
#error messured by the mean of the square differnces between predictions
# and targets
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train,y_train,epochs=100,batch_size=32)

#Making predictions-Validation step

#getting stock prices of 2017
dataset_test=pd.read_csv('/home/kostas/Desktop/Udemy/excersice 3/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

#prediction-getting the predicted stock price of 2017
#concatenation required because we use 60 timesets...test set has only 20 days
#without the prevous days of which exists on training set we can not predict
#anything
dataset_total=pd.concat((dataset_training['Open'],dataset_test['Open']),axis=0)
#this has a result=1198(the 60 days we need from the training set)
#basically we have the 60 days we need from the training set + 20 days from test set
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1) #add a new axis to make it suitable for kkeras
inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualising results
plt.plot(real_stock_price,color='red',label='real google stock prices')
plt.plot(predicted_stock_price,color='blue',label='predicted by rnn')
plt.title('Google stock proce prediction')
plt.legend()
plt.show()


