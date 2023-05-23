#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
numArgs = len(sys.argv)
# You want 2 variables, and the 0th index will be '[file].py'.
if (numArgs >= 11):
    SGE_TASK_ID = sys.argv[1]
    print(SGE_TASK_ID)
    runs = int(sys.argv[2])
    print(runs)
    train_percent=float(sys.argv[3])
    val_percent=float(sys.argv[4]) 
    print(train_percent, val_percent)
    posnoise_percent=float(sys.argv[5])
    print(posnoise_percent)
    negnoise_percent=float(sys.argv[6])
    print(negnoise_percent)
    w0_degree, w1_degree, w2_degree, b_degree = float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10])
    print(w0_degree, w1_degree, w2_degree, b_degree)
    job_id = sys.argv[11]
    print(job_id)
    
       
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


# # DATA generation

# In[8]:


def prod(items, start=1):
    for item in items:
        start *= item
    return start


def perm(n, k):
    if not 0 <= k <= n:
        raise ValueError(
            'Values must be non-negative and n >= k in perm(n, k)')
    else:
        return prod(range(n - k + 1, n + 1))


def comb(n, k):
    if not 0 <= k <= n:
        raise ValueError(
            'Values must be non-negative and n >= k in comb(n, k)')
    else:
        k = k if k < n - k else n - k
        return prod(range(n - k + 1, n + 1)) // math.factorial(k)


# In[9]:


def create_network_new(no_nodes, w0_degree, w1_degree, w2_degree, b_degree, w_lower, w_upper, b_lower, b_upper, 
                   imbalance_nodes_percent1, imbalance_nodes_percent2, within_noise0=0, within_noise1=0, within_noise2=0, between_noise=0):
    random.seed(2411)
    np.random.seed(2411)
    G = nx.gnp_random_graph(int(no_nodes*imbalance_nodes_percent1),1,random.randint(0,10000))
    G = nx.relabel_nodes(G, lambda x: 'g-'+str(x))
    labels_g = [0]
    color_g='black'
    nx.set_node_attributes(G, labels_g, "labels")
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.uniform(w_lower, w_upper)
        G.edges[u,v]['color'] = color_g
    H = nx.gnp_random_graph(int(no_nodes*imbalance_nodes_percent2),1,random.randint(0,10000))
    H = nx.relabel_nodes(H, lambda x: 'h-'+str(x))
    labels_h = [1]
    color_h='black'
    nx.set_node_attributes(H, labels_h, "labels")
    for (u, v) in H.edges():
        H.edges[u,v]['weight'] = random.uniform(w_lower, w_upper)
        H.edges[u,v]['color'] = color_h
    J = nx.gnp_random_graph(int(no_nodes*(1-imbalance_nodes_percent1-imbalance_nodes_percent2)),1,random.randint(0,10000))
    J = nx.relabel_nodes(J, lambda x: 'j-'+str(x))
    labels_j = [2]
    color_j='black'
    nx.set_node_attributes(J, labels_j, "labels")
    for (u, v) in J.edges():
        J.edges[u,v]['weight'] = random.uniform(w_lower, w_upper)
        J.edges[u,v]['color'] = color_j
    
    K = nx.union(G,H)
    K = nx.union(K,J)
    edges = list(K.edges)
    nonedges = list(nx.non_edges(K))
    chosen_nonedges_ls = []
    
    w0_edges_ratio=w0_degree*len(G.nodes)/perm(len(G.nodes),2)
    w1_edges_ratio=w1_degree*len(H.nodes)/perm(len(H.nodes),2)
    w2_edges_ratio=w2_degree*len(J.nodes)/perm(len(J.nodes),2)
    b_edges_ratio=b_degree*no_nodes/(len(G.nodes)*len(H.nodes)*2+len(G.nodes)*len(J.nodes)*2+len(J.nodes)*len(H.nodes)*2)
    
    edges0 = list(G.edges)
    edges_select0 = edges0.copy()
    
    
    start = timeit.default_timer()
    chosen_edge0_set = random.sample(edges_select0, int(len(edges0)-len(edges0)*w0_edges_ratio))
    
    G.remove_edges_from(chosen_edge0_set)
    
    edges_select0 = list(G.edges()).copy()
        
    stop = timeit.default_timer()
     
    within_noise_edges0 = []
    within_noise_indices0 = np.random.choice(len(edges_select0), int(len(edges_select0)*within_noise0), replace=False)
    
    within_noise_edges0 = [edges_select0[i] for i in within_noise_indices0]
    
    for (u, v) in within_noise_edges0:
        G.edges[u,v]['weight'] = random.uniform(b_lower, b_upper)
    
    edges1 = list(H.edges)
    edges_select1 = edges1.copy()
    
    start = timeit.default_timer()
    chosen_edge1_set = random.sample(edges_select1, int(len(edges1)-len(edges1)*w1_edges_ratio))
    
    H.remove_edges_from(chosen_edge1_set)
    edges_select1 = list(H.edges()).copy()
        
    within_noise_edges1 = []
    within_noise_indices1 = np.random.choice(len(edges_select1), int(len(edges_select1)*within_noise1), replace=False)
    within_noise_edges1 = [edges_select1[i] for i in within_noise_indices1]

    for (u, v) in within_noise_edges1:
        H.edges[u,v]['weight'] = random.uniform(b_lower, b_upper)
    
   
    
    #J within degree
    edges2 = list(J.edges)
    edges_select2 = edges2.copy()
    chosen_edge2_set = random.sample(edges_select2, int(len(edges2)-len(edges2)*w2_edges_ratio))
    
    J.remove_edges_from(chosen_edge2_set)
    edges_select2 = list(J.edges()).copy()
        

    within_noise_edges2 = []
    within_noise_indices2 = np.random.choice(len(edges_select2), int(len(edges_select2)*within_noise2), replace=False)
    within_noise_edges2 = [edges_select2[i] for i in within_noise_indices2]
    for (u, v) in within_noise_edges2:
        J.edges[u,v]['weight'] = random.uniform(b_lower, b_upper)
    
   
    K = nx.union(G,H)
    K = nx.union(K,J)
    #between-community edges
    nonedges_select = nonedges.copy()   
    chosen_nonedges_ls = random.sample(nonedges_select, int(len(nonedges)*b_edges_ratio))
    
    K.add_edges_from(chosen_nonedges_ls)
    for (u, v) in chosen_nonedges_ls:
        K.edges[u,v]['weight'] = random.uniform(b_lower, b_upper)
        
    between_noise_indices = np.random.choice(len(chosen_nonedges_ls), int(len(chosen_nonedges_ls)*between_noise), replace=False)
    
    between_noise_edges = [chosen_nonedges_ls[i] for i in between_noise_indices]
    
    for (u, v) in between_noise_edges:
        K.edges[u,v]['weight'] = random.uniform(w_lower, w_upper)
    
    return G, H, J, K


# In[10]:


def create_feature(G, H, J, mean_gf1, sd_gf1, mean_hf1, sd_hf1, mean_jf1, sd_jf1, 
                   mean_gf2, sd_gf2, mean_hf2, sd_hf2, mean_jf2, sd_jf2, 
                   lower_gf1, upper_gf1, lower_hf1, upper_hf1, lower_jf1, upper_jf1, 
                   lower_gf2, upper_gf2, lower_hf2, upper_hf2, lower_jf2, upper_jf2,):
    random.seed(2411)
    np.random.seed(2411)
    f10 = get_truncated_normal(mean=mean_gf1, sd=sd_gf1, low=lower_gf1, upp=upper_gf1)
    f11 = get_truncated_normal(mean=mean_hf1, sd=sd_hf1, low=lower_hf1, upp=upper_hf1)
    f12 = get_truncated_normal(mean=mean_jf1, sd=sd_jf1, low=lower_jf1, upp=upper_jf1)
    f20 = get_truncated_normal(mean=mean_gf2, sd=sd_gf2, low=lower_gf2, upp=upper_gf2)
    f21 = get_truncated_normal(mean=mean_hf2, sd=sd_hf2, low=lower_hf2, upp=upper_hf2)
    f22 = get_truncated_normal(mean=mean_jf2, sd=sd_jf2, low=lower_jf2, upp=upper_jf2)
    data1 = {'node': G.nodes, 
             'features1': f10.rvs(len(G.nodes)),
             'features2': f20.rvs(len(G.nodes))}

    df1 = pd.DataFrame(data1)
    df1['label']=0
    df1['node'] =  df1['node'].astype(str)

    data2 = {'node': H.nodes, 
             'features1': f11.rvs(len(H.nodes)),
             'features2': f21.rvs(len(H.nodes))}

    df2 = pd.DataFrame(data2)
    df2['label']=1
    df2['node'] =  df2['node'].astype(str)
    
    data3 = {'node': J.nodes, 
             'features1': f12.rvs(len(J.nodes)),
             'features2': f22.rvs(len(J.nodes))}

    df3 = pd.DataFrame(data3)
    df3['label']=2
    df3['node'] =  df3['node'].astype(str)
    df = pd.concat([df1, df2, df3])
    return df


# In[11]:


def data_split(df, train_percent, val_percent):
    random.seed(int(SGE_TASK_ID))
    np.random.seed(int(SGE_TASK_ID))
    nodes = list(range(no_nodes))
    nodes0 = df.mapping[:int(len(df)*imbalance_nodes_percent1)].values
    nodes1 = df.mapping[int(len(df)*imbalance_nodes_percent1):int(len(df)*(imbalance_nodes_percent1+imbalance_nodes_percent2))].values
    nodes2 = df.mapping[int(len(df)*(imbalance_nodes_percent1+imbalance_nodes_percent2)):].values

    data_node0=pd.DataFrame({'node': nodes0, 'class': [0] * len(nodes0)})
    data_node1=pd.DataFrame({'node': nodes1, 'class': [1] * len(nodes1)})
    data_node2=pd.DataFrame({'node': nodes2, 'class': [2] * len(nodes2)})
    data_node= pd.DataFrame(pd.concat([data_node0, data_node1, data_node2]))
    data_node = data_node.reset_index(drop=True)

    train_node_set = []
    val_node_set = []
    test_node_set = []
    
    train_nodes_dt=data_node.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=train_percent))
    train_nodes=train_nodes_dt.node.values
    
    data_val_diff=pd.concat([data_node, train_nodes_dt, train_nodes_dt]).drop_duplicates(keep=False)
    val_nodes_dt=data_val_diff.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=val_percent*len(data_node)/len(data_val_diff)))
    val_nodes=val_nodes_dt.node.values

    data_test_diff=pd.concat([data_val_diff, val_nodes_dt, val_nodes_dt]).drop_duplicates(keep=False)
    test_nodes_dt=data_test_diff.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=(1.0-(train_percent+val_percent))*len(data_node)/len(data_test_diff)))
    test_nodes=test_nodes_dt.node.values

    train_node_set.append(train_nodes)
    val_node_set.append(val_nodes)
    test_node_set.append(test_nodes)
    
    train_masks = torch.zeros(no_nodes, dtype=torch.bool)
    val_masks = torch.zeros(no_nodes, dtype=torch.bool)
    test_masks = torch.zeros(no_nodes, dtype=torch.bool)
    for i in train_node_set:
        train_masks[i] = True

    for i in val_node_set:
        val_masks[i] = True

    for i in test_node_set:
        test_masks[i] = True
    return train_masks, val_masks, test_masks, train_node_set, val_node_set, test_node_set


# In[12]:


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


# In[13]:


from scipy.stats import truncnorm

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# # GCN

# In[14]:


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


# In[15]:


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

# In[16]:


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


# In[17]:


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

# In[18]:


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


# In[19]:


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

# In[20]:


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


# In[21]:


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

# In[22]:


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


# In[23]:


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

# In[24]:


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


# In[25]:


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

# In[26]:


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


# In[27]:


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

# In[28]:


# SGE_TASK_ID = 9
# runs =1

# train_percent=0.1
# val_percent=0.3
# posnoise_percent=0
# negnoise_percent=0

# w0_degree = 3
# w1_degree = 3
# w2_degree = 3
# b_degree = 3

# job_id = 1


# In[29]:


no_nodes = 300
mean_gf1, sd_gf1, lower_gf1, upper_gf1 = 60, 20, 0, 120
mean_hf1, sd_hf1, lower_hf1, upper_hf1= 80, 20, 0, 120
mean_jf1, sd_jf1, lower_jf1, upper_jf1= 100, 20, 0, 120

mean_gf2, sd_gf2, lower_gf2, upper_gf2 = 25, 20, 0, 100
mean_hf2, sd_hf2, lower_hf2, upper_hf2= 35, 20, 0, 100
mean_jf2, sd_jf2, lower_jf2, upper_jf2= 45, 20, 0, 100


w_lower, w_upper, b_lower, b_upper = 1, 1, -1, -1

imbalance_nodes_percent1 = 1/3
imbalance_nodes_percent2 = 1/3

start = timeit.default_timer()
features, targets, edge_index, edge_weight, edge_weight_nor, df = network_inputs()
train_masks, val_masks, test_masks, train_nodes, val_nodes, test_nodes = data_split(df, train_percent, val_percent)
stop = timeit.default_timer()
print('TOTAL Time: ', stop - start)  
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(features[train_masks])
feats_norm = np.zeros(features.shape)
feats_norm[train_masks]=min_max_scaler.transform(features[train_masks])
feats_norm[val_masks]=min_max_scaler.transform(features[val_masks])
feats_norm[test_masks]=min_max_scaler.transform(features[test_masks])
feats_norm=torch.from_numpy(feats_norm.astype(np.float32))

standard_scaler = preprocessing.StandardScaler()
standard_scaler.fit(features[train_masks])
feats_standard = np.zeros(features.shape)
feats_standard[train_masks]=standard_scaler.transform(features[train_masks])
feats_standard[val_masks]=standard_scaler.transform(features[val_masks])
feats_standard[test_masks]=standard_scaler.transform(features[test_masks])
feats_standard=torch.from_numpy(feats_standard.astype(np.float32))


# In[30]:


edge_type=torch.zeros(edge_index.shape[1])
edge_type[np.where(edge_weight<0)]=1
num_relations=np.unique(edge_type).shape[0]
data = Data(x=features, edge_index=edge_index,y=targets,edge_weight=edge_weight,
               train_mask=train_masks, val_mask=val_masks, test_mask=test_masks, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data}')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')


# In[31]:


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


# In[32]:


import itertools
def com_class_func(x,y):
    com = [x,y]
    combination = [p for p in itertools.product(*com)]
    return combination
from itertools import combinations
  
def rSubset(arr, r):
    return list(combinations(arr, r))


# In[33]:


class_list=[class0,class1,class2]
class_comb=rSubset(class_list, 2)
neg_comb=[]
for i in range(len(class_comb)):
    comb=com_class_func(class_comb[i][0],class_comb[i][1])
    neg_comb.append(comb)
neg_comb = [item for sublist in neg_comb for item in sublist]

neg_comb_arr=np.asarray(neg_comb).T
neg_comb_arr1=neg_comb_arr[0]
neg_comb_arr2=neg_comb_arr[1]
neg_comb_arr12=np.append([neg_comb_arr1], [neg_comb_arr2])
neg_comb_arr21=np.append([neg_comb_arr2], [neg_comb_arr1])
neg_comb_arr = np.append([neg_comb_arr12], [neg_comb_arr21], axis=0)
neg_comb_tensor=torch.LongTensor(neg_comb_arr)


# In[34]:


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


# In[35]:


pos_comb_c0=pos_comb_function(class0)
pos_comb_c1=pos_comb_function(class1)
pos_comb_c2=pos_comb_function(class2)
pos_comb=pos_comb_c0+pos_comb_c1+pos_comb_c2
pos_comb_arr=np.asarray(pos_comb).T
pos_comb_arr1=pos_comb_arr[0]
pos_comb_arr2=pos_comb_arr[1]
pos_comb_arr12=np.append([pos_comb_arr1], [pos_comb_arr2])
pos_comb_arr21=np.append([pos_comb_arr2], [pos_comb_arr1])
pos_comb_arr = np.append([pos_comb_arr12], [pos_comb_arr21], axis=0)
pos_comb_tensor=torch.LongTensor(pos_comb_arr)


# In[36]:


#check whether negative edges have the same pair as positive edges:

def find_overlap(A, B):

    if not A.dtype == B.dtype:
        raise TypeError("A and B must have the same dtype")
    if not A.shape[1:] == B.shape[1:]:
        raise ValueError("the shapes of A and B must be identical apart from "
                         "the row dimension")

    A = np.ascontiguousarray(A.reshape(A.shape[0], -1))
    B = np.ascontiguousarray(B.reshape(B.shape[0], -1))

    t = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))

    return np.in1d(A.view(t), B.view(t))


# In[37]:


edge_index_G={tuple(e) for e in np.asarray(data.edge_index).T}
G=nx.from_edgelist(sorted(edge_index_G))
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())
print('Edges are directed:', nx.is_directed(G))
print('Number of self-loops:', nx.number_of_selfloops(G))
print('Number of isolated nodes:', len(list(nx.isolates(G))))


# In[38]:


#find how many actual negative links in original links
ori_poslink_idx =np.where(find_overlap(data.edge_index.T,neg_comb_tensor.T) == False)[0]
print("number of true pos links in original links:", len(ori_poslink_idx))
ori_poslink_idx_1way =np.where(find_overlap(torch.tensor(list(G.edges())),neg_comb_tensor.T) == False)[0]
print("number of true pos links in original links (1way):", len(ori_poslink_idx_1way))


# In[39]:


#find how many actual negative links in original links
ori_neglink_idx =np.where(find_overlap(data.edge_index.T,neg_comb_tensor.T) == True)[0]
print("number of true neg links in original links:", len(ori_neglink_idx))
ori_neglink_idx_1way =np.where(find_overlap(torch.tensor(list(G.edges())),neg_comb_tensor.T) == True)[0]
print("number of true neg links in original links (1way):", len(ori_neglink_idx_1way))


# In[40]:


ori_pos_link_1way=[list(G.edges())[i] for i in ori_poslink_idx_1way]
ori_neg_link_1way=[list(G.edges())[i] for i in ori_neglink_idx_1way]


# In[41]:


#fixing positive links, adding noise to negative links

pos_link_tuple=[tuple(i) for i in np.array(data.edge_index[:,ori_poslink_idx]).T]
mutual=set(pos_comb).intersection(set(pos_link_tuple))
unused_pos_comb=list(set(pos_comb)-mutual)

#fixing negative links, adding noise to positive links

neg_link_tuple=[tuple(i) for i in np.array(data.edge_index[:,ori_neglink_idx]).T]
mutual=set(neg_comb).intersection(set(neg_link_tuple))
unused_neg_comb=list(set(neg_comb)-mutual)


# In[42]:


random.seed(int(SGE_TASK_ID))
if posnoise_percent>0:
    poslink_noise_1way=random.sample(unused_pos_comb, k=int(posnoise_percent/2*len(ori_neglink_idx)))
    remove_neglink=random.sample(ori_neg_link_1way, k=int(posnoise_percent/2*len(ori_neglink_idx))) 
    remain_true_neglink=list(set(ori_neg_link_1way)-set(remove_neglink))
    new_neglink=np.append(remain_true_neglink,poslink_noise_1way, axis=0)
    edge_row1 = np.append(np.array([edge for edge in np.array(new_neglink)]).T[0], np.array([edge for edge in np.array(new_neglink)]).T[1])
    edge_row2 = np.append(np.array([edge for edge in np.array(new_neglink)]).T[1], np.array([edge for edge in np.array(new_neglink)]).T[0])
    new_neglink_2way = np.append([edge_row1], [edge_row2], axis=0)
    new_neglink_2way=torch.LongTensor(new_neglink_2way)
    
else:
    new_neglink_2way=data.edge_index[:,np.where(data.edge_weight<0)[0]]




if negnoise_percent>0:
    neglink_noise_1way=random.sample(unused_neg_comb, k=int(negnoise_percent/2*len(ori_poslink_idx)))
    remove_poslink=random.sample(ori_pos_link_1way, k=int(negnoise_percent/2*len(ori_poslink_idx))) 
    remain_true_poslink=list(set(ori_pos_link_1way)-set(remove_poslink))
    new_poslink=np.append(remain_true_poslink,neglink_noise_1way, axis=0)
    edge_row1 = np.append(np.array([edge for edge in np.array(new_poslink)]).T[0], np.array([edge for edge in np.array(new_poslink)]).T[1])
    edge_row2 = np.append(np.array([edge for edge in np.array(new_poslink)]).T[1], np.array([edge for edge in np.array(new_poslink)]).T[0])
    new_poslink_2way = np.append([edge_row1], [edge_row2], axis=0)
    new_poslink_2way=torch.LongTensor(new_poslink_2way)
    
else:
    new_poslink_2way=data.edge_index[:,np.where(data.edge_weight>0)[0]]
    
edge_index_new=torch.cat((new_neglink_2way,new_poslink_2way),1)
edge_weight_new = torch.zeros(edge_index_new.shape[1])
edge_weight_new[:new_neglink_2way.shape[1]] = -1
edge_weight_new[new_neglink_2way.shape[1]:] = 1


# In[43]:


edge_type=torch.zeros(edge_index_new.shape[1])
edge_type[np.where(edge_weight_new<0)]=1
num_relations=np.unique(edge_type).shape[0]
data_new = Data(x=features, edge_index=edge_index_new,y=targets,edge_weight=edge_weight_new,
               train_mask=train_masks, val_mask=val_masks, test_mask=test_masks, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_new}')
print(f'Edges are directed: {data_new.is_directed()}')
print(f'Graph has isolated nodes: {data_new.has_isolated_nodes()}')
print(f'Graph has loops: {data_new.has_self_loops()}')


# In[44]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_pos= Data(x=features, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight>0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_pos}')
print(f'Edges are directed: {data_pos.is_directed()}')
print(f'Graph has isolated nodes: {data_pos.has_isolated_nodes()}')
print(f'Graph has loops: {data_pos.has_self_loops()}')
print(f'Number of relations: {data_pos.num_relations}')


# In[45]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_neg= Data(x=data_new.x, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight<0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_neg}')
print(f'Edges are directed: {data_neg.is_directed()}')
print(f'Graph has isolated nodes: {data_neg.has_isolated_nodes()}')
print(f'Graph has loops: {data_neg.has_self_loops()}')
print(f'Number of relations: {data_neg.num_relations}')


# In[46]:


edge_type=torch.zeros(edge_index_new.shape[1])
edge_type[np.where(edge_weight_new<0)]=1
num_relations=np.unique(edge_type).shape[0]
data_norm = Data(x=feats_norm, edge_index=data_new.edge_index,y=targets,edge_weight=data_new.edge_weight,
               train_mask=train_masks, val_mask=val_masks, test_mask=test_masks, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_norm}')
print(f'Edges are directed: {data_norm.is_directed()}')
print(f'Graph has isolated nodes: {data_norm.has_isolated_nodes()}')
print(f'Graph has loops: {data_norm.has_self_loops()}')


# In[47]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_pos_norm = Data(x=feats_norm, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight>0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_pos_norm}')
print(f'Edges are directed: {data_pos_norm.is_directed()}')
print(f'Graph has isolated nodes: {data_pos_norm.has_isolated_nodes()}')
print(f'Graph has loops: {data_pos_norm.has_self_loops()}')
print(f'Number of relations: {data_pos_norm.num_relations}')


# In[48]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_neg_norm= Data(x=feats_norm, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight<0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_neg_norm}')
print(f'Edges are directed: {data_neg_norm.is_directed()}')
print(f'Graph has isolated nodes: {data_neg_norm.has_isolated_nodes()}')
print(f'Graph has loops: {data_neg_norm.has_self_loops()}')
print(f'Number of relations: {data_neg_norm.num_relations}')


# In[49]:


edge_type=torch.zeros(edge_index_new.shape[1])
edge_type[np.where(edge_weight_new<0)]=1
num_relations=np.unique(edge_type).shape[0]
data_standard = Data(x=feats_standard, edge_index=data_new.edge_index,y=targets,edge_weight=data_new.edge_weight,
               train_mask=train_masks, val_mask=val_masks, test_mask=test_masks, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_norm}')
print(f'Edges are directed: {data_norm.is_directed()}')
print(f'Graph has isolated nodes: {data_norm.has_isolated_nodes()}')
print(f'Graph has loops: {data_norm.has_self_loops()}')


# In[50]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_pos_standard= Data(x=feats_standard, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight>0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight>0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_pos_standard}')
print(f'Edges are directed: {data_pos_standard.is_directed()}')
print(f'Graph has isolated nodes: {data_pos_standard.has_isolated_nodes()}')
print(f'Graph has loops: {data_pos_standard.has_self_loops()}')
print(f'Number of relations: {data_pos_standard.num_relations}')


# In[51]:


edge_type=torch.zeros(data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]].shape[1])
num_relations=np.unique(edge_type).shape[0]

data_neg_standard= Data(x=feats_standard, edge_index=data_new.edge_index[:,np.where(data_new.edge_weight<0)[0]],y=data_new.y,edge_weight=data_new.edge_weight[np.where(data_new.edge_weight<0)[0]],
               train_mask=data_new.train_mask, val_mask=data_new.val_mask, test_mask=data_new.test_mask, 
               edge_type=edge_type, num_relations=num_relations)
print(f'Graph: {data_neg_standard}')
print(f'Edges are directed: {data_neg_standard.is_directed()}')
print(f'Graph has isolated nodes: {data_neg_standard.has_isolated_nodes()}')
print(f'Graph has loops: {data_neg_standard.has_self_loops()}')
print(f'Number of relations: {data_neg_standard.num_relations}')


# # Running

# In[52]:


def save_data( gcn, gcn_pos, rgcn_net, gcn_lap, 
              gcn_diff, gcn_nlinear, gcn_diff_split, gcn_nlinear_split, 
              data_type):
    data={'gcn': gcn, 'gcn_pos': gcn_pos, 'rgcn_net':rgcn_net, 
          'gcn_lap':gcn_lap,'gcn_diff': gcn_diff, 'gcn_nlinear': gcn_nlinear, 
          'gcn_diff_split': gcn_diff_split, 'gcn_nlinear_split': gcn_nlinear_split, 
         }
    data = pd.DataFrame([data])
    data.to_csv(cwd + '/job_id{0}/'.format(job_id)+data_type+'{0}.csv'.format(SGE_TASK_ID))


# In[53]:


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


# In[54]:


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


# In[55]:


lr=0.01
p,e=50,301

acc_gcn, f1_micro_gcn, f1_macro_gcn, f1_weighted_gcn=gcn_function(data_norm,runs,p,e)
acc_gcn_pos, f1_micro_gcn_pos, f1_macro_gcn_pos, f1_weighted_gcn_pos=gcn_function(data_pos_norm,runs,p,e)
acc_rgcn_net, f1_micro_rgcn_net, f1_macro_rgcn_net, f1_weighted_rgcn_net=rgcn_net_function(data_norm,runs,p,e)
acc_gcnlap, f1_micro_gcnlap, f1_macro_gcnlap, f1_weighted_gcnlap=gcnlap_function(data_norm,runs,p,e)
acc_gcndiff, f1_micro_gcndiff, f1_macro_gcndiff, f1_weighted_gcndiff=gcndiff_function(data_norm,runs,p,e)
acc_gcn_nlinear, f1_micro_gcn_nlinear, f1_macro_gcn_nlinear, f1_weighted_gcn_nlinear=gcn_nonlinear_function(data_norm,runs,p,e)
acc_gcndiff_split, f1_micro_gcndiff_split, f1_macro_gcndiff_split, f1_weighted_gcndiff_split=gcndiff_split_function(data_norm,runs,p,e)
acc_gcn_nlinear_split, f1_micro_gcn_nlinear_split, f1_macro_gcn_nlinear_split, f1_weighted_gcn_nlinear_split=gcn_nonlinear_split_function(data_norm,runs,p,e)


# In[56]:


cwd = os.getcwd()
save_data( acc_gcn, acc_gcn_pos, acc_rgcn_net,
           acc_gcnlap, acc_gcndiff, acc_gcn_nlinear,
           acc_gcndiff_split, acc_gcn_nlinear_split,
          'acc_norm')
cwd = os.getcwd()
save_data( f1_micro_gcn, f1_micro_gcn_pos, f1_micro_rgcn_net, 
           f1_micro_gcnlap, f1_micro_gcndiff, f1_micro_gcn_nlinear, 
           f1_micro_gcndiff_split, f1_micro_gcn_nlinear_split,
          'f1_micro_norm')
cwd = os.getcwd()
save_data( f1_macro_gcn, f1_macro_gcn_pos, f1_macro_rgcn_net, 
           f1_macro_gcnlap, f1_macro_gcndiff, f1_macro_gcn_nlinear, 
           f1_macro_gcndiff_split, f1_macro_gcn_nlinear_split, 
          'f1_macro_norm')
cwd = os.getcwd()
save_data( f1_weighted_gcn, f1_weighted_gcn_pos, f1_weighted_rgcn_net, 
           f1_weighted_gcnlap, f1_weighted_gcndiff, f1_weighted_gcn_nlinear, 
           f1_weighted_gcndiff_split, f1_weighted_gcn_nlinear_split, 
          'f1_weighted_norm')


# In[57]:


lr=0.01
p,e=50,301

acc_gcn, f1_micro_gcn, f1_macro_gcn, f1_weighted_gcn=gcn_function(data_standard,runs,p,e)
acc_gcn_pos, f1_micro_gcn_pos, f1_macro_gcn_pos, f1_weighted_gcn_pos=gcn_function(data_pos_standard,runs,p,e)
acc_rgcn_net, f1_micro_rgcn_net, f1_macro_rgcn_net, f1_weighted_rgcn_net=rgcn_net_function(data_standard,runs,p,e)
acc_gcnlap, f1_micro_gcnlap, f1_macro_gcnlap, f1_weighted_gcnlap=gcnlap_function(data_standard,runs,p,e)
acc_gcndiff, f1_micro_gcndiff, f1_macro_gcndiff, f1_weighted_gcndiff=gcndiff_function(data_standard,runs,p,e)
acc_gcn_nlinear, f1_micro_gcn_nlinear, f1_macro_gcn_nlinear, f1_weighted_gcn_nlinear=gcn_nonlinear_function(data_standard,runs,p,e)
acc_gcndiff_split, f1_micro_gcndiff_split, f1_macro_gcndiff_split, f1_weighted_gcndiff_split=gcndiff_split_function(data_standard,runs,p,e)
acc_gcn_nlinear_split, f1_micro_gcn_nlinear_split, f1_macro_gcn_nlinear_split, f1_weighted_gcn_nlinear_split=gcn_nonlinear_split_function(data_standard,runs,p,e)


# In[58]:


cwd = os.getcwd()
save_data( acc_gcn, acc_gcn_pos, acc_rgcn_net,
           acc_gcnlap, acc_gcndiff, acc_gcn_nlinear,
           acc_gcndiff_split, acc_gcn_nlinear_split,
          'acc_standard')
cwd = os.getcwd()
save_data( f1_micro_gcn, f1_micro_gcn_pos, f1_micro_rgcn_net, 
           f1_micro_gcnlap, f1_micro_gcndiff, f1_micro_gcn_nlinear, 
           f1_micro_gcndiff_split, f1_micro_gcn_nlinear_split,
          'f1_micro_standard')
cwd = os.getcwd()
save_data( f1_macro_gcn, f1_macro_gcn_pos, f1_macro_rgcn_net, 
           f1_macro_gcnlap, f1_macro_gcndiff, f1_macro_gcn_nlinear, 
           f1_macro_gcndiff_split, f1_macro_gcn_nlinear_split, 
          'f1_macro_standard')
cwd = os.getcwd()
save_data( f1_weighted_gcn, f1_weighted_gcn_pos, f1_weighted_rgcn_net, 
           f1_weighted_gcnlap, f1_weighted_gcndiff, f1_weighted_gcn_nlinear, 
           f1_weighted_gcndiff_split, f1_weighted_gcn_nlinear_split, 
          'f1_weighted_standard')


# In[ ]:




