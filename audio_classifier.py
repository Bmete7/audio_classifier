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



class DatasetCreation(nn.Module):
    
    def __init__(self, data_size = 64, dataset_length = 100 ):
        super(DatasetCreation, self).__init__()
        
        self.data = np.zeros((dataset_length, data_size) , dtype =np.float64)
        self.labels = np.zeros((dataset_length, 1) , dtype =np.int32)
        if(math.log(data_size) % 1 != 0):
            assert('shape mismatch')
        self.n_qubits = int(math.log(data_size))
        for i in range(dataset_length):
            class_label = np.random.randint(2)
            self.labels[i] = class_label
            if(class_label  == 0):
                initial_value = np.random.random_sample() / 2 
                self.data[i, 0] = initial_value
                increase_amount = (np.random.random_sample() / 20)
                decrease_amount = (np.random.random_sample() / 30)
                for j in range(1, data_size):
                    if( j <= data_size / 2):
                        self.data[i, j] = self.data[i,j-1] + increase_amount
                    else:
                        self.data[i, j] = self.data[i,j-1] - decrease_amount
                
            elif(class_label  == 1):
                initial_value = (np.random.random_sample() + 1) / 2 
                self.data[i, 0] = initial_value
                increase_amount = (np.random.random_sample() / 30)
                decrease_amount = (np.random.random_sample() / 20)
                for j in range(1, data_size):
                    if( j <= data_size / 2):
                        self.data[i, j] = self.data[i,j-1] - decrease_amount
                    else:
                        self.data[i, j] = self.data[i,j-1] + increase_amount
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr,inputs = False):
            
            
        @qml.template
        def AmplitudeEmbedding(self,inputs):
            qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(0,), normalize = True,pad=(0.j))
            
        weight_shapes = {"weights_r": (2 , self.n_qubits, 3),"weights_cr": (self.n_qubits,self.n_qubits-1 ,3)}
        
        # weights_st = torch.tensor(qml.init.strong_ent_layers_uniform(3, self.training_qubits_size), requires_grad=True)
        
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
            
d = DatasetCreation(16,100)


d.data
d.labels



