#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:04:28 2019
#https://skymind.ai/wiki/restricted-boltzmann-machine
@author: kostas
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#the 2nd parameter the seperator between varaibles
#header is actually the line from which we fetch the headers's names
movies=pd.read_csv('ml-1m/movies.dat',sep='::', header=None , engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::', header=None , engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::', header=None , engine='python',encoding='latin-1')

#preparing the training set and test set
training_set=pd.read_csv('ml-100k/u1.base',header=None,delimiter='\t')
training_set=np.array(training_set,dtype='int')

test_set=pd.read_csv('ml-100k/u1.test',header=None,delimiter='\t')
test_set=np.array(test_set,dtype='int')

#getting the number of users and movies
nb_users=int(max(max(training_set[:,0]) , max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]) , max(test_set[:,1])))

#converting the data into an array with users in lines and movies in columns
#RBM expects the features and we have some observations going one by one
#into the network starting with the input nodes and what we actually have to do 
#is create a straucture that will contain these pbservations that will go into
#the network 

#we will have  the observations in lines and the feature in collumns
#the usual structure for a common ANN

def convert(data):
     new_data=[]
     for id_users in range(1,nb_users+1):
         #get all the movies ud for the current user
         #we need the second condition to get the movies_ids for this specific user
         id_movies=data[:,1][data[:,0]==id_users]
         id_ratings=data[:,2][data[:,0]==id_users]
         ratings=np.zeros(nb_movies)
         #the indexes ids of movies starts at 1 
         #in python idexes starts at 0. So we need the movies ids to start the
         #same base as the indexes of the list
         #in that way we have the movies rated by the specific user
         #if movie not rated then we have rating=0-watch line 52
         ratings[id_movies-1]=id_ratings
         #we need list of lists because of torch
         new_data.append(list(ratings))
     return new_data
#training set 
training_set=convert(training_set)
test_set=convert(test_set)

#PYTORCH TENSORS
#converting data to Torch tensors(for building the architecture)
#tensors=arrays that contain elements of a single data type
#multidimensional array instead of being numpy array this is a torch array
 
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

#converting ratings into binary ratings for the needs of recstricted bolzman machine
#1->Liked , 0->disliked
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1

test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1

#Create the architecture of RBM
#RBM is an energy based functionn an we have an energy function which 
#we try to minimize.We actually try to adjust the weights to get the wanted
#minimization.It can also be handled as pobablistic model
#we want to calculate the gradient.we move on to the gradient so we can
#minimize the energy function
#Bias is the constant term in the linear coefficients.
#y = mx+c (c is the bias here)
class RBM:
    def __init__(self,nv,nh):
        #Randomly initialized
        #init weights
        self.W=torch.randn(nh,nv)
        #init bias>. The bias value allows the activation
        #function to be shifted to the left or right, to better fit the data.
        #https://medium.com/deeper-learning/glossary-of-deep-learning-bias-cf49d9c895e2
        self.a=torch.randn(1,nh)#we need at least 2d tensor
        #bias for visible nodes
        self.b=torch.randn(1,nv)
    #Sample the hidden nodes according to the probalities "vh" where
    #v=visible nodes and h=hidden nodes.It's the sigmoid activation function
    #Samples function is actually for gibbs sampling
    def sample_h(self,x):
        #multiply two arrays as fas far it concerns W i need the transposal
        #because of dimensions of two arrays
        wx=torch.mm(x,self.W.t())
        #we expand the dimensions of bias from hidden layer in order to be added
        #to everey element of wx
        activation=wx+self.a.expand_as(wx)
        #calculate the probability of getting "activation" given "visible"
        #bayes probability
        ph_given_v=torch.sigmoid(activation)
        return ph_given_v,torch.bernoulli(ph_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        pv_given_h=torch.sigmoid(activation)
        #we actually get if the movie was liked or not
        return pv_given_h,torch.bernoulli(pv_given_h)
    def train(self,v0,vk,ph0,phk):    
    #v0=all the ratings by one user
    #vk=visible nodes obtained after k samplings
    #ph0=probabilities of the hidden nodes given the ratings of user(v0)
    #phk=probabilities after k-sampling given the values of visible nodes
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)

#number of visible nodes.The number of movies in a row
nv=len(training_set[0])
#number of hidden nodes..corresponds to the number of features we want to detect
nh=100
#update the weights after batch size
batch_size=100

rbm=RBM(nv,nh)

nb_epoch=10
for epoch in range(1,nb_epoch+1):
     train_loss=0
     s=0.
     #update the weights after each batch of users going to the network
     #e.g->0-99---99-199 until the end
     for id_user in range(0,nb_users-batch_size,batch_size):
         #vk=the ratings of the users for the current batch
         vk=training_set[id_user:id_user+batch_size]
         #the targerts at the very beggining has the same value with vk(input)
         #however the input is reconstructed and we need to keep the targets
         v0=training_set[id_user:id_user+batch_size]
         #probability of hidden nodes given the ratings of movies rated by the users of batch size
         #ignore the 2nd variable returned-we keep the real probalities too
         #just like we did with targets/inputs
         ph0,_=rbm.sample_h(vk)
         #CD procedure with k steps
         for k in range(10):
             #input the ratings of users
             _,hk=rbm.sample_h(vk)
             _,vk=rbm.sample_v(hk)
             #we do not the rbm to be trained to the cells that have -1
             #-1 has the cells which the movies has not been rated 
             #we actually need this for the line 171
             vk[v0<0]=v0[v0<0]
             #print(vk)
         phk,_=rbm.sample_h(vk) 
         rbm.train(v0,vk,ph0,phk)
         #Loss function:simple distance in absolute values between the prediction
         #and the real rating----Mean distance between the predicted-real ratings
         train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
         s+=1.
     print('epoch:'+str(epoch)+' loss:'+str(train_loss/s))
         
     
#testing (Blind walk principle similary to random walk but the probabilities
#are not the same)
test_loss=0
s=0.
for id_user in range(nb_users):
    v=training_set[id_user:id_user+1] 
    vt=test_set[id_user:id_user+1]
    #we want only the movies which have been rated
    if len(vt[vt>=0]>0):
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        print(str(vt.size())+'----'+str(v.size()))
        test_loss+=torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s+=1.
print('Test loss:'+str(test_loss/s))
