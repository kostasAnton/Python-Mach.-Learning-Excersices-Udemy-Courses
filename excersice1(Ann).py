# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#excersice 1 by udemy
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/kostas/Desktop/Udemy/Exercise 1/Artificial_Neural_Networks/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Now we need to take care tha categorail variables 
#The variable gender for example takes values Male or Female
#we need to convert it to 0 or 1. The same thing stands for Geography too.
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Geography = LabelEncoder()
X[:, 1] = labelencoder_Geography.fit_transform(X[:, 1])

labelencoder_Gender = LabelEncoder()
X[:, 2] = labelencoder_Gender.fit_transform(X[:, 2])

#In order to remove dummy variables.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#we have two of the three countries in our array
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#installing an ANN
classifier = Sequential()
#we have just added the input layer+the first hidden layer
classifier.add(Dense(output_dim=6 , init='uniform', activation='relu', input_dim=11))
#another hidden layer
classifier.add(Dense(output_dim=6 , init='uniform', activation='relu'))
#outout layer
classifier.add(Dense(output_dim=1 , init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

classifier.fit(X_train , y_train , batch_size=10 , nb_epoch=100)

y_predict=classifier.predict(X_test)
#a natural threshold
y_predict=(y_predict>0.5)

from sklearn.metrics import confusion_matrix
confusioMatrix=confusion_matrix(y_predict,y_test)


#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
newPrediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
newPrediction=(newPrediction>0.5)


##Evaluating and improving tunig ANN  - PART 4
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def construct_classifier():
    classifier = Sequential()
    #we have just added the input layer+the first hidden layer
    classifier.add(Dense(output_dim=6 , init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))
    #another hidden layer
    classifier.add(Dense(output_dim=6 , init='uniform', activation='relu'))
    #outout layer
    classifier.add(Dense(output_dim=1 , init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])
    return classifier

kerasClassifier=KerasClassifier(build_fn=construct_classifier , batch_size=10 ,epochs=100)
accuracies=cross_val_score(estimator=kerasClassifier , X=X_train, y=y_train , cv=10 , n_jobs=1)
#if we observe the accuracies we will see that there a big variance between our values
#as a result our ANN has been overfitted. Ovefit:When there is big variance between 
#the accuracies on kFolds and when the variance between the test set and the training set
#has a high value.
#in that case we need the drop outs
mean=accuracies.mean()
variance=accuracies.std()


#Tuning the ANN-PART 4 ..A study on how we can tune our ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def construct_classifier():
    classifier = Sequential()
    #we have just added the input layer+the first hidden layer
    classifier.add(Dense(output_dim=6 , init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))
    #another hidden layer
    classifier.add(Dense(output_dim=6 , init='uniform', activation='relu'))
    #outout layer
    classifier.add(Dense(output_dim=1 , init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])
    return classifier

kerasClassifier=KerasClassifier(build_fn=construct_classifier)

#Now we are going to check different hyper parameters the accuracy they are giving us
#build a dictionairy with the parameters
parameters={'batch_size':[25,32],
            'epochs':[100,500],
            #'optimizer':['adam','rmsprop']
            }

gridSearch=GridSearchCV(estimator=kerasClassifier,
                        param_grid=parameters , 
                        scoring='accuracy',cv=10,n_jobs=1)

gridSearch=gridSearch.fit(X_train,y_train)
bestParameters=gridSearch.best_params_
bestAccuracy=gridSearch.best_score_