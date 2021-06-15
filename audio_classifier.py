# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:56:40 2021

@author: burak
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from copy import deepcopy
import math

import time
import timeit
import numpy as np
import torch
from torch.autograd import Function

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import pennylane as qml

import matplotlib.pyplot as plt


import timeit
from torch.utils.tensorboard import SummaryWriter



import torchvision



class DatasetCreation(nn.Module):
    
                        
    def __init__(self,dev , data_size = 64, dataset_length = 100 ):
        super(DatasetCreation, self).__init__()
        
        
        self.split = 5/4
        self.train_size = int(dataset_length /self.split)
        self.data = np.zeros((self.train_size, data_size) , dtype =np.float64)
        self.test_data = np.zeros((dataset_length - self.train_size , data_size) , dtype =np.float64)
        self.labels = np.zeros((self.train_size, 1) , dtype =np.int32)
        self.test_labels = np.zeros((dataset_length - self.train_size, 1) , dtype =np.int32)
        
        self.data_size = data_size
        self.dataset_length = dataset_length
        
        if(math.log(data_size) % 1 != 0):
            assert('shape mismatch')
        # self.n_qubits = int(math.log2(data_size))
        self.n_qubits = 8
        self.create_data()
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr,inputs = False):
            # self.AmplitudeEmbedding(inputs)
            self.embedding(inputs)
            
            # Exactly the same as the MPS at the Stoudenmire paper. 
            # Add images in the AWS notebook
            for i in range(0, self.n_qubits):
                qml.Rot(*weights_r[0, i], wires = i)
            for i in range(0,self.n_qubits-1):
                qml.CRot( *weights_cr[0,i] , wires= [i,i+1])
            for i in range(0, self.n_qubits):
                qml.Rot(*weights_r[1, i], wires = i)                    
            for i in range(0,self.n_qubits-1):
                qml.CRot( *weights_cr[1,i] , wires= [i,i+1])    
            for i in range(0, self.n_qubits):
                qml.Rot(*weights_r[2, i], wires = i)
                
            return qml.expval(qml.PauliZ(self.n_qubits - 1))
            
        
            
        weight_shapes = {"weights_r": (3 , self.n_qubits, 3),"weights_cr": (2, self.n_qubits-1,3)}
        
        # weights_st = torch.tensor(qml.init.strong_ent_layers_uniform(3, self.training_qubits_size), requires_grad=True)
        
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        
    @qml.template
    def embedding(self,inputs):
        qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,self.n_qubits), rotation = 'X')
        
    @qml.template
    def AmplitudeEmbedding(self,inputs):
        qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(0,self.n_qubits), normalize = True,pad=(0.j))
    def create_data(self):
        for i in range(self.dataset_length):
            class_label = np.random.randint(2)
            if(i >= self.train_size):
                self.test_labels[i - self.train_size ] = class_label
            else:
                self.labels[i  ] = class_label
            
            if(class_label  == 0):
                initial_value = np.random.random_sample() / 2
                if(i >= self.train_size):
                    self.test_data[i - self.train_size, 0] = initial_value    
                else:
                    self.data[i, 0] = initial_value
                increase_amount = (np.random.random_sample() / 20)
                decrease_amount = (np.random.random_sample() / 30)
                
                increase_amount = (1 / 20)
                decrease_amount = (1 / 30)
                
                for j in range(1, self.data_size):
                    increase_amount = (np.random.random_sample() / 30)
                    decrease_amount = (np.random.random_sample() / 20)
                    if( j <= self.data_size / 2):
                        if(i >= self.train_size):
                            self.test_data[i - self.train_size, j] = self.test_data[i - self.train_size,j-1] + increase_amount
                        else:
                            self.data[i, j] = self.data[i,j-1] + increase_amount
                    else:
                        if(i >= self.train_size):
                            self.test_data[i - self.train_size, j] = self.test_data[i- self.train_size,j-1] - decrease_amount
                        else:
                            self.data[i, j] = self.data[i,j-1] - decrease_amount
                
            elif(class_label  == 1):
                initial_value = (np.random.random_sample() + 1) / 2 
                
                if(i >= self.dataset_length / self.split):
                    self.test_data[i - self.train_size, 0] = initial_value    
                else:
                    self.data[i, 0] = initial_value
                
                increase_amount = (np.random.random_sample() / 30)
                decrease_amount = (np.random.random_sample() / 20)
                increase_amount = (1 / 30)
                decrease_amount = (1 / 20)
                for j in range(1, self.data_size):
                    increase_amount = (np.random.random_sample() / 30)
                    decrease_amount = (np.random.random_sample() / 20)
                    if( j <= self.data_size / 2):
                        if(i >= self.train_size):
                            self.test_data[i - self.train_size, j] = self.test_data[i- self.train_size,j-1] - decrease_amount
                        else:
                            self.data[i, j] = self.data[i,j-1] - decrease_amount
                    else:
                        if(i >= self.train_size):
                            self.test_data[i - self.train_size, j] = self.test_data[i - self.train_size,j-1] + increase_amount
                        else:
                            self.data[i, j] = self.data[i,j-1] + increase_amount
    def forward(self,x):
        
        x =  self.qlayer(x)
        
        return x
    def getData(self):
        return self.data,self.test_data,self.labels,self.test_labels

dev = qml.device("default.qubit", wires=8 ,shots = 1000)
d = DatasetCreation(dev,8,50)


dataset,test_dataset,labels,test_labels = d.getData()

def Fidelity_loss(mes,label):
    '''tot  =0
    print(mes , ' + ')
    for i in mes[0]:
        tot += i[0]
        print(tot)
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    
    print(fidelity)
    mes -= 0.00000001
    return torch.log(1- mes)'''
    
    return ((mes+ 1 ) / 2 - label) ** 2


learning_rate = 0.05
epochs = 9
loss_list = []

# opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
opt = torch.optim.Adam(d.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss


test_loss = nn.MSELoss()

# %%  The Training part
opt.zero_grad()
for epoch in range(epochs):
    opt.zero_grad()
    total_loss = []
    start_time = timeit.time.time()
    for i in range(len(labels)):
        
        # for iris dataseti
        # data = datas['data']
        
        data = dataset[i]
        label = labels[i]
        
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But we do it anyways for visualization
        
        
        normalized = torch.Tensor(data).view(1,-1)
        label = torch.Tensor(label).view(1,-1)
         
        out = d(normalized)
        loss = loss_func(out,label)
        loss.backward()
        
        if(i%10 == 0):
            print(loss)
            opt.step()
            opt.zero_grad()
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))

