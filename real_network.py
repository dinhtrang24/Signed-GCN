#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
numArgs = len(sys.argv)
# You want 2 variables, and the 0th index will be '[file].py'.
if (numArgs >= 5):
    SGE_TASK_ID = sys.argv[1]
    print(SGE_TASK_ID)
    runs = int(sys.argv[2])
    print(runs)
    posnoise_percent=float(sys.argv[3])
    print(posnoise_percent)
    negnoise_percent=float(sys.argv[4])
    print(negnoise_percent)
    feats_noise_percent=float(sys.argv[5])
    print(feats_noise_percent)
     
    
else:
    print('Not enough arguments.')


# In[2]:


import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter_add


# In[3]:


import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
from datetime import date, timedelta
import os
import random
import requests
import datetime
#%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.linalg import fractional_matrix_power
# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import copy

import time
import timeit
from tqdm import tqdm
pd.set_option('display.float_format', lambda x: '%.6f' % x)

import warnings
warnings.filterwarnings("ignore")


# In[4]:


import sys
sys.path.insert(0, os.getcwd())
from signed_models import GCNConv_lap, GCNConv_diff, GCNConv_nonlinear, GCNConv_diff_split, GCNConv_nonlinear_split


# In[5]:


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold


# In[6]:


#define Earlystopping as Pytorch does support it automatically
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


# In[7]:


def accuracy(pred_y, y):
            return (pred_y == y).sum() / len(y)


# # Import data

# In[8]:


from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# In[9]:


def network_inputs():
    random.seed(2411)
    np.random.seed(2411)
    start = timeit.default_timer()
 
    G, H, J, K=create_network_new(no_nodes, w0_degree, w1_degree, w2_degree, b_degree, w_lower, w_upper, b_lower, b_upper, 
                   imbalance_nodes_percent1, imbalance_nodes_percent2, 0,0,0,0)
    stop = timeit.default_timer()
     
    start = timeit.default_timer()
    df = create_feature(G, H, J, mean_gf1, sd_gf1, mean_hf1, sd_hf1, mean_jf1, sd_jf1, 
                   mean_gf2, sd_gf2, mean_hf2, sd_hf2, mean_jf2, sd_jf2, 
                   lower_gf1, upper_gf1, lower_hf1, upper_hf1, lower_jf1, upper_jf1, 
                   lower_gf2, upper_gf2, lower_hf2, upper_hf2, lower_jf2, upper_jf2,)
    stop = timeit.default_timer()
    
    mapping = dict(zip(K, range(no_nodes)))
    M = nx.relabel_nodes(K, mapping)
    
    df['mapping'] = mapping.values()
    features = torch.from_numpy((np.array(df.loc[:, ~df.columns.isin(['node', 'label', 'mapping'])])).astype(np.float32))
    targets = df['label'].values
    targets = torch.LongTensor(targets)
    if len(list(M.edges()))==0:
        edge_index = torch.LongTensor(np.zeros((2,1)))
        edge_weight = torch.zeros(1)
        edge_weight_nor = torch.zeros(1)
    else: 
        

        edge_row1 = np.append(np.array([edge for edge in M.edges()]).T[0], np.array([edge for edge in M.edges()]).T[1])
        edge_row2 = np.append(np.array([edge for edge in M.edges()]).T[1], np.array([edge for edge in M.edges()]).T[0])
        edge = np.append([edge_row1], [edge_row2], axis=0)
        edge_index = torch.LongTensor(edge)
        
        edge_weight = []
        for e in K.edges:
            w = K[e[0]][e[1]]['weight']
            edge_weight.append(w)
        edge_weight = np.stack(edge_weight)
        edge_weight = np.append([edge_weight], [edge_weight])
        edge_weight = torch.from_numpy((np.array(edge_weight)).astype(np.float32))
        edge_weight_nor = edge_weight.clone().detach()
        edge_weight_nor -= edge_weight_nor.min()
        edge_weight_nor /= edge_weight_nor.max()
    return features, targets, edge_index, edge_weight, edge_weight_nor, df


# # GCN

# In[10]:


from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv(data_new.x.shape[1], 16)
        self.out = GCNConv(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index).relu()
        z = self.out(h, edge_index)
        return h, z


# In[11]:


def gcn_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    val_losses_ls=[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN()
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.bias.data.fill_(0)
        model.out.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val (model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
            
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break

            embeddings.append(h)
            losses.append(loss)
            val_losses.append(val_loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
        val_losses_ls.append(val_losses)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls, 


# # GCN-LAP

# In[12]:


class GCN_lap(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv_lap(data_new.x.shape[1], 16)
        self.out = GCNConv_lap(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index, edge_weight):
        h = self.gcn1(x, edge_index, edge_weight).relu()
        z = self.out(h, edge_index, edge_weight)
        return h, z


# In[13]:


def gcnlap_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    val_losses_ls=[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN_lap()
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.lin.bias.data.fill_(0)
        model.out.lin.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val (model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index, data.edge_weight)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
            
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break

            embeddings.append(h)
            losses.append(loss)
            val_losses.append(val_loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {best_test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
        val_losses_ls.append(val_losses)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls


# # GCN-diff

# In[14]:


class GCN_diff(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv_diff(data_new.x.shape[1], 16)
        self.out = GCNConv_diff(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index, edge_weight):
        h = self.gcn1(x, edge_index, edge_weight).relu()
        z = self.out(h, edge_index, edge_weight)
        return h, z


# In[15]:


def gcndiff_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    val_losses_ls=[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN_diff()
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.lin.bias.data.fill_(0)
        model.out.lin.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val (model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index, data.edge_weight)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break

            embeddings.append(h)
            losses.append(loss)
            val_losses.append(val_loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {best_test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
        val_losses_ls.append(val_losses)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls, 


# # GCN-nonlinear

# In[16]:


class GCN_nonlinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv_nonlinear(data_new.x.shape[1], 16)
        self.out = GCNConv_nonlinear(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index, edge_weight):
        h = self.gcn1(x, edge_index, edge_weight).relu()
        z = self.out(h, edge_index, edge_weight)
        return h, z


# In[17]:


def gcn_nonlinear_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    val_losses_ls=[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN_nonlinear() 
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.lin.bias.data.fill_(0)
        model.out.lin.bias.data.fill_(0)
    
        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.

            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val(model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index, data.edge_weight)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
                
#                 best_test_acc = test_acc
#                 best_test_f1 = test_f1
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break
            embeddings.append(h)
            losses.append(loss)
            val_losses.append(val_loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {best_test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
        val_losses_ls.append(val_losses)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls


# # RGCN

# In[18]:


from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.nn import FastRGCNConv, RGCNConv
class RGCN_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = RGCNConv(data_new.x.shape[1], 16, data_new.num_relations)
        self.out = RGCNConv(16, len(np.unique(data_new.y)), data_new.num_relations)
                              

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.gcn1(x, edge_index, edge_type))
        z = self.out(h, edge_index, edge_type)
        return h,z


# In[19]:


def rgcn_net_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    val_losses_ls=[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = RGCN_Net()
        torch.nn.init.xavier_uniform(model.gcn1.weight)
        torch.nn.init.xavier_uniform(model.out.weight)
        model.gcn1.bias.data.fill_(0)
        model.out.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_type)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.

            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val(model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x,data.edge_index, data.edge_type)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x,data.edge_index, data.edge_type)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)

            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break
            embeddings.append(h)
            losses.append(loss)
            val_losses.append(val_loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
        val_losses_ls.append(val_losses)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls


# # GCN-diff-split

# In[20]:


class GCN_diff_split(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv_diff_split(data_new.x.shape[1], 16)
        self.out = GCNConv_diff_split(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index, edge_weight):
        h = self.gcn1(x, edge_index, edge_weight).relu()
        z = self.out(h, edge_index, edge_weight)
        return h, z


# In[21]:


def gcndiff_split_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN_diff_split()
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.lin.bias.data.fill_(0)
        model.out.lin.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val (model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index, data.edge_weight)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            losses.append(loss)
            val_losses.append(val_loss)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
                
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls


# # GCN-nonlinear-split

# In[22]:


class GCN_nonlinear_split(torch.nn.Module):
    def __init__(self):
        super().__init__()
#         torch.manual_seed(12345)
        self.gcn1 = GCNConv_nonlinear_split(data_new.x.shape[1], 16)
        self.out = GCNConv_nonlinear_split(16, len(np.unique(data_new.y)))

    def forward(self, x, edge_index, edge_weight):
        h = self.gcn1(x, edge_index, edge_weight).relu()
        z = self.out(h, edge_index, edge_weight)
        return h, z


# In[23]:


def gcn_nonlinear_split_function(data, t, p, e): 
    embeddings_ls, losses_ls, accuracies_ls, outputs_ls = [],[],[],[]
    test_acc_ls, test_f1_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls  = [],[],[],[],[]
    for i in range(t):
        print('RUN:', i)
        model = GCN_nonlinear_split() 
        torch.nn.init.xavier_uniform(model.gcn1.lin.weight)
        torch.nn.init.xavier_uniform(model.out.lin.weight)
        model.gcn1.lin.bias.data.fill_(0)
        model.out.lin.bias.data.fill_(0)

        # Data for animations
        embeddings = []
        losses = []
        accuracies = []
        outputs = []
        
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        es = EarlyStopping(patience=p)

        def train(model):
            model.train()   # setting the model to train mode
            optimizer.zero_grad()  # Clear gradients.
            h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.

            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()  # Update parameters based on gradients.
            return loss,h,out

        def val(model):
            with torch.no_grad():
                model.eval()
                h,out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the validation nodes.
                return val_loss

        def test(model):
            with torch.no_grad():   #Turn off gradients.
                model.eval()   # set model to evaluation mode
                h,out = model(data.x, data.edge_index, data.edge_weight)
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
                test_f1_micro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='micro')
                test_f1_macro = f1_score(data.y[data.test_mask], pred[data.test_mask], average='macro')
                test_f1_weighted = f1_score(data.y[data.test_mask], pred[data.test_mask], average='weighted')
                cm = confusion_matrix(data.y[data.test_mask], pred[data.test_mask])
                return test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm

        losses, val_losses = [], []
        min_val_loss = 1000000
        for epoch in range(1, e):
            loss,h,out = train(model)
            val_loss = val(model)
            test_acc, test_f1_micro, test_f1_macro, test_f1_weighted, cm = test(model)
            losses.append(loss)
            val_losses.append(val_loss)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        
            print('Epoch:', epoch, ' Train loss:', loss, ' Val loss:', val_loss, 'Min Val loss:', min_val_loss, 'test acc:', test_acc)
            if es.step(val_loss):
                break
            embeddings.append(h)
            losses.append(loss)
            accuracies.append(test_acc)
            outputs.append(out.argmax(dim=1))
        best_test_acc, best_test_f1_micro, best_test_f1_macro, best_test_f1_weighted, best_cm = test(best_model)
        
        test_acc_ls.append(best_test_acc)
        test_f1_micro_ls.append(best_test_f1_micro)
        test_f1_macro_ls.append(best_test_f1_macro)
        test_f1_weighted_ls.append(best_test_f1_weighted)
        print(f'Test Accuracy: {test_acc:.5f}')
        model.eval()   # set model to evaluation mode
        embeddings_ls.append(embeddings)
        accuracies_ls.append(accuracies)
        losses_ls.append(losses)
        outputs_ls.append(outputs)
    return test_acc_ls, test_f1_micro_ls, test_f1_macro_ls, test_f1_weighted_ls


# # Assign signs to original links

# In[24]:


class0, class1, class2, class3, class4, class5, class6=[],[],[],[],[],[],[]
for i in range(len(data.y)):
    if data.y[i]==0:
        class0.append(i)
    elif data.y[i]==1:
        class1.append(i)
    elif data.y[i]==2:
        class2.append(i)
    elif data.y[i]==3:
        class3.append(i)
    elif data.y[i]==4:
        class4.append(i)
    elif data.y[i]==5:
        class5.append(i)
    else:
        class6.append(i)


# In[25]:


import itertools
def com_class_func(x,y):
    com = [x,y]
    combination = [p for p in itertools.product(*com)]
    return combination
from itertools import combinations
  
def rSubset(arr, r):
    return list(combinations(arr, r))


# In[26]:


class_list=[class0,class1,class2,class3,class4,class5,class6]
class_comb=rSubset(class_list, 2)
neg_comb=[]
for i in range(len(class_comb)):
    comb=com_class_func(class_comb[i][0],class_comb[i][1])
    neg_comb.append(comb)
neg_comb = [item for sublist in neg_comb for item in sublist]


# In[27]:


neg_comb_arr=np.asarray(neg_comb).T
neg_comb_arr1=neg_comb_arr[0]
neg_comb_arr2=neg_comb_arr[1]
neg_comb_arr12=np.append([neg_comb_arr1], [neg_comb_arr2])
neg_comb_arr21=np.append([neg_comb_arr2], [neg_comb_arr1])
neg_comb_arr = np.append([neg_comb_arr12], [neg_comb_arr21], axis=0)
neg_comb_tensor=torch.LongTensor(neg_comb_arr)


# In[28]:


def pos_comb_function(within_class):
    class_list=[within_class, within_class]
    class_comb=rSubset(class_list, 2)
    pos_comb_ls=[]
    for i in range(len(class_comb)):
        comb=com_class_func(class_comb[i][0],class_comb[i][1])
        pos_comb_ls.append(comb)
    pos_comb_ls = [item for sublist in pos_comb_ls for item in sublist]
    pos_comb_ls=[pos_comb_ls[i] for i in range(len(pos_comb_ls)) if pos_comb_ls[i][0] != pos_comb_ls[i][1] ]
    
    return pos_comb_ls


# In[29]:


pos_comb_c0=[]
pos_comb_c0=pos_comb_function(class0)
pos_comb_c1=pos_comb_function(class1)
pos_comb_c2=pos_comb_function(class2)
pos_comb_c3=pos_comb_function(class3)
pos_comb_c4=pos_comb_function(class4)
pos_comb_c5=pos_comb_function(class5)
pos_comb_c6=pos_comb_function(class6)

pos_comb=pos_comb_c0+pos_comb_c1+pos_comb_c2+pos_comb_c3+pos_comb_c4+pos_comb_c5+pos_comb_c6


# In[30]:


pos_comb_arr=np.asarray(pos_comb).T
pos_comb_arr1=pos_comb_arr[0]
pos_comb_arr2=pos_comb_arr[1]
pos_comb_arr12=np.append([pos_comb_arr1], [pos_comb_arr2])
pos_comb_arr21=np.append([pos_comb_arr2], [pos_comb_arr1])
pos_comb_arr = np.append([pos_comb_arr12], [pos_comb_arr21], axis=0)
pos_comb_tensor=torch.LongTensor(pos_comb_arr)


# In[31]:


#check whether negative edges have the same pair as positive edges:

def find_overlap(A, B):

    if not A.dtype == B.dtype:
        raise TypeError("A and B must have the same dtype")
    if not A.shape[1:] == B.shape[1:]:
        raise ValueError("the shapes of A and B must be identical apart from "
                         "the row dimension")

    # reshape A and B to 2D arrays. force a copy if neccessary in order to
    # ensure that they are C-contiguous.
    A = np.ascontiguousarray(A.reshape(A.shape[0], -1))
    B = np.ascontiguousarray(B.reshape(B.shape[0], -1))

    # void type that views each row in A and B as a single item
    t = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))

    # use in1d to find rows in A that are also in B
    return np.in1d(A.view(t), B.view(t))


# In[32]:


edge_index={tuple(e) for e in np.asarray(data.edge_index).T}
G=nx.from_edgelist(sorted(edge_index))
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())
print('Edges are directed:', nx.is_directed(G))
print('Number of self-loops:', nx.number_of_selfloops(G))
print('Number of isolated nodes:', len(list(nx.isolates(G))))


# In[33]:


#find how many actual negative links in original links
ori_poslink_idx =np.where(find_overlap(data.edge_index.T,neg_comb_tensor.T) == False)[0]
print("number of true pos links in original links:", len(ori_poslink_idx))
ori_poslink_idx_1way =np.where(find_overlap(torch.tensor(list(G.edges())),neg_comb_tensor.T) == False)[0]
print("number of true pos links in original links (1way):", len(ori_poslink_idx_1way))


# In[34]:


#find how many actual negative links in original links
ori_neglink_idx =np.where(find_overlap(data.edge_index.T,neg_comb_tensor.T) == True)[0]
print("number of true neg links in original links:", len(ori_neglink_idx))
ori_neglink_idx_1way =np.where(find_overlap(torch.tensor(list(G.edges())),neg_comb_tensor.T) == True)[0]
print("number of true pos links in original links (1way):", len(ori_neglink_idx_1way))


# In[35]:


ori_pos_link_1way=[list(G.edges())[i] for i in ori_poslink_idx_1way]
ori_neg_link_1way=[list(G.edges())[i] for i in ori_neglink_idx_1way]


# In[36]:


# SGE_TASK_ID=2
# random.seed(int(SGE_TASK_ID))
# posnoise_percent=0
# negnoise_percent=0
# runs=1
# feats_noise_percent=0.0


# In[37]:


#fixing positive links, adding noise to negative links

pos_link_tuple=[tuple(i) for i in np.array(data.edge_index[:,ori_poslink_idx]).T]
mutual=set(pos_comb).intersection(set(pos_link_tuple))
unused_pos_comb=list(set(pos_comb)-mutual)

#fixing negative links, adding noise to positive links

neg_link_tuple=[tuple(i) for i in np.array(data.edge_index[:,ori_neglink_idx]).T]
mutual=set(neg_comb).intersection(set(neg_link_tuple))
unused_neg_comb=list(set(neg_comb)-mutual)


# In[38]:


random.seed(int(SGE_TASK_ID))
if posnoise_percent>0:
    poslink_noise_1way=random.sample(unused_pos_comb, k=int(posnoise_percent/2*len(ori_neglink_idx)))
    edge_row1 = np.append(np.array([edge for edge in np.array(poslink_noise_1way)]).T[0], np.array([edge for edge in np.array(poslink_noise_1way)]).T[1])
    edge_row2 = np.append(np.array([edge for edge in np.array(poslink_noise_1way)]).T[1], np.array([edge for edge in np.array(poslink_noise_1way)]).T[0])
    poslink_noise_2way = np.append([edge_row1], [edge_row2], axis=0)
    poslink_noise_2way=torch.LongTensor(poslink_noise_2way)
    
#     ori_poslink_noise_idx=np.where(find_overlap(data.edge_index.T,poslink_noise_2way.T) == True)[0]
else:
    poslink_noise_2way=torch.LongTensor([])


if negnoise_percent>0:
    neglink_noise_1way=random.sample(unused_neg_comb, k=int(negnoise_percent/2*len(ori_poslink_idx)))
    edge_row1 = np.append(np.array([edge for edge in np.array(neglink_noise_1way)]).T[0], np.array([edge for edge in np.array(neglink_noise_1way)]).T[1])
    edge_row2 = np.append(np.array([edge for edge in np.array(neglink_noise_1way)]).T[1], np.array([edge for edge in np.array(neglink_noise_1way)]).T[0])
    neglink_noise_2way = np.append([edge_row1], [edge_row2], axis=0)
    neglink_noise_2way=torch.LongTensor(neglink_noise_2way)
#     ori_neglink_noise_idx=np.where(find_overlap(data.edge_index.T,neglink_noise_2way.T) == True)[0]
else:
    neglink_noise_2way=torch.LongTensor([])
    
edge_index_new=torch.cat((data.edge_index,poslink_noise_2way),1)
edge_index_new=torch.cat((edge_index_new,neglink_noise_2way),1)
# edge_index_new=edge_index_new.long()
edge_weight_new = torch.zeros(edge_index_new.shape[1])
edge_weight_new[ori_poslink_idx] = 1
edge_weight_new[ori_neglink_idx] = -1

if posnoise_percent>0:
    poslink_noise_idx=np.where(find_overlap(edge_index_new.T,poslink_noise_2way.T) == True)[0]
    edge_weight_new[poslink_noise_idx] = -1

if negnoise_percent>0:
    neglink_noise_idx=np.where(find_overlap(edge_index_new.T,neglink_noise_2way.T) == True)[0]
    edge_weight_new[neglink_noise_idx] = 1


print('no. of negative links:', sum(edge_weight_new==-1))
print('no. of positive links:', sum(edge_weight_new==1))
print('percent of negative links:', np.round(sum(edge_weight_new==-1)/len(edge_weight_new)*100,2), "%") 
print('percent of positive links:', np.round(sum(edge_weight_new==1)/edge_weight_new.shape[0]*100,2), "%")


# In[39]:


noisy_feats=data.x.clone()
feat_noise_idx_ls=[]
for i in range(noisy_feats.shape[0]):
    random_feats=np.random.randint(2, size=(int(feats_noise_percent*noisy_feats.shape[1])))
    random_feats=torch.from_numpy(random_feats.astype(np.float32))
    feat_noise_idx=random.sample(range(noisy_feats.shape[1]), k=int(feats_noise_percent*noisy_feats.shape[1]))
    feat_noise_idx_ls.append(feat_noise_idx)
    noisy_feats[i][feat_noise_idx]=random_feats


# In[40]:


edge_type=torch.zeros(edge_index_new.shape[1])
edge_type[np.where(edge_weight_new==-1)]=1
num_relations=np.unique(edge_type).shape[0]
data_new = Data(x=noisy_feats, edge_index=edge_index_new,y=data.y,edge_weight=edge_weight_new,
               train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_new}')
print(f'Edges are directed: {data_new.is_directed()}')
print(f'Graph has isolated nodes: {data_new.has_isolated_nodes()}')
print(f'Graph has loops: {data_new.has_self_loops()}')


# In[41]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight==1)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_pos= Data(x=data_new.x, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight==1)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight==1)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_pos}')
print(f'Edges are directed: {data_pos.is_directed()}')
print(f'Graph has isolated nodes: {data_pos.has_isolated_nodes()}')
print(f'Graph has loops: {data_pos.has_self_loops()}')
print(f'Number of relations: {data_pos.num_relations}')


# In[42]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight==-1)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_neg= Data(x=data_new.x, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight==-1)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight==-1)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_neg}')
print(f'Edges are directed: {data_neg.is_directed()}')
print(f'Graph has isolated nodes: {data_neg.has_isolated_nodes()}')
print(f'Graph has loops: {data_neg.has_self_loops()}')
print(f'Number of relations: {data_pos.num_relations}')


# # Running

# In[43]:


def save_data( gcn, gcn_pos, rgcn_net, gcn_lap, 
              gcn_diff, gcn_nlinear, gcn_diff_split, gcn_nlinear_split, 
              data_type):
    data={'gcn': gcn, 'gcn_pos': gcn_pos, 'rgcn_net':rgcn_net, 
          'gcn_lap':gcn_lap,'gcn_diff': gcn_diff, 'gcn_nlinear': gcn_nlinear, 
          'gcn_diff_split': gcn_diff_split, 'gcn_nlinear_split': gcn_nlinear_split, 
         }
    data = pd.DataFrame([data])
    data.to_csv(cwd + '/result/'+data_type+'{0}.csv'.format(SGE_TASK_ID)) 


# In[44]:


lr=0.01
p,e=50,301

acc_gcn, f1_micro_gcn, f1_macro_gcn, f1_weighted_gcn=gcn_function(data_new,runs,p,e)
acc_gcn_pos, f1_micro_gcn_pos, f1_macro_gcn_pos, f1_weighted_gcn_pos=gcn_function(data_pos,runs,p,e)
acc_rgcn_net, f1_micro_rgcn_net, f1_macro_rgcn_net, f1_weighted_rgcn_net=rgcn_net_function(data_new,runs,p,e)
acc_gcnlap, f1_micro_gcnlap, f1_macro_gcnlap, f1_weighted_gcnlap=gcnlap_function(data_new,runs,p,e)
acc_gcndiff, f1_micro_gcndiff, f1_macro_gcndiff, f1_weighted_gcndiff=gcndiff_function(data_new,runs,p,e)
acc_gcn_nlinear, f1_micro_gcn_nlinear, f1_macro_gcn_nlinear, f1_weighted_gcn_nlinear=gcn_nonlinear_function(data_new,runs,p,e)
acc_gcndiff_split, f1_micro_gcndiff_split, f1_macro_gcndiff_split, f1_weighted_gcndiff_split=gcndiff_split_function(data_new,runs,p,e)
acc_gcn_nlinear_split, f1_micro_gcn_nlinear_split, f1_macro_gcn_nlinear_split, f1_weighted_gcn_nlinear_split=gcn_nonlinear_split_function(data_new,runs,p,e)


# In[45]:


cwd = os.getcwd()
save_data( acc_gcn, acc_gcn_pos, acc_rgcn_net,
           acc_gcnlap, acc_gcndiff, acc_gcn_nlinear,
           acc_gcndiff_split, acc_gcn_nlinear_split,
          'acc')
cwd = os.getcwd()
save_data( f1_micro_gcn, f1_micro_gcn_pos, f1_micro_rgcn_net, 
           f1_micro_gcnlap, f1_micro_gcndiff, f1_micro_gcn_nlinear, 
           f1_micro_gcndiff_split, f1_micro_gcn_nlinear_split,
          'f1_micro')
cwd = os.getcwd()
save_data( f1_macro_gcn, f1_macro_gcn_pos, f1_macro_rgcn_net, 
           f1_macro_gcnlap, f1_macro_gcndiff, f1_macro_gcn_nlinear, 
           f1_macro_gcndiff_split, f1_macro_gcn_nlinear_split, 
          'f1_macro')
cwd = os.getcwd()
save_data( f1_weighted_gcn, f1_weighted_gcn_pos, f1_weighted_rgcn_net, 
           f1_weighted_gcnlap, f1_weighted_gcndiff, f1_weighted_gcn_nlinear, 
           f1_weighted_gcndiff_split, f1_weighted_gcn_nlinear_split, 
          'f1_weighted')


# In[ ]:




