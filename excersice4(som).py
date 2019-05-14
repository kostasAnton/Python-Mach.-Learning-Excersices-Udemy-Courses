#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:15:53 2019

@author: kostas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#fraud detection

#importing my dataset (usi machine learning repository)
 dataset=pd.read_csv('/home/kostas/Desktop/Udemy/excersice 4/Credit_Card_Applications.csv')
 #all the collumns expect the last one
 #making the split in order to seperate the customers who approved between those who didn't
 X= dataset.iloc[:,:-1].values
 #only the last collumn
 y= dataset.iloc[:,-1].values

#Feature  scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

from minisom import MiniSom
som=MiniSom(x=10, y=10 , input_len=15,sigma=1.0,learning_rate=0.5 ) 
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100) 

from pylab import bone, pcolor,colorbar,plot,show
bone()#in order to initialize the window
#get the mean neuron distance between the nodes(winning nodes)(neighbourhood of nodes)
#by the distance we can identify the frauds because if there is high
#flactuation then we may have a fraud
#in the graph we can say that it is formed neighbourhoods according to the color
#For example if we have a very different in a neighbourhood which is based 
#on black then we have a possible fraud
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
#detection of id for frauds
for i,x in enumerate(X):
    #think the winning node as the visual represantation of a customer in a 2D plot
    #get the winning node for current customer
    w=som.winner(x)
    #visualization
    #w[1] and w[2] the coordinates of the winning node ..add 0.5 to place
    #the marker att the center
    plot(w[0]+0.5,w[1]+0.5,
         markers[y[i]],markeredgecolor=colors[y[i]],
         markersize=10,
         markeredgewidth=2,
         markerfacecolor='None')
show()    
 
#get the winning nodes for the customers
#each item of dictionary is the winning node with customers associates on its
mappings=som.win_map(X)
frauds=mappings[(5,5)]
frauds=sc.inverse_transform(frauds)
 
 
 
 
 