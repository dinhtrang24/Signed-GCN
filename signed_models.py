#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add


class GCNConv_lap(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_lap, self).__init__(aggr='add') 
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index_self_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_weight_self_loop = torch.cat((edge_weight, torch.ones(x.size(0))), dim=0)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index_self_loop
        deg = scatter_add(torch.abs(edge_weight_self_loop), col, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * torch.abs(edge_weight_self_loop) * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index_self_loop, x=x, norm=norm, edge_weight_self_loop=edge_weight_self_loop)
    def message(self, x_i, x_j, norm, edge_weight_self_loop):

        x_j = torch.sign(edge_weight_self_loop).view(-1,1)*x_j 

        # Step 4: Normalize node features.
        return (norm.view(-1, 1) * x_j)
    

class GCNConv_diff(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv_diff, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index_self_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight_self_loop = torch.cat((edge_weight, torch.ones(x.size(0))), dim=0)

            # Step 2: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 3: Compute normalization.
            row, col = edge_index_self_loop
            deg = scatter_add(torch.abs(edge_weight_self_loop), col, dim=0, dim_size=x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * torch.abs(edge_weight_self_loop) * deg_inv_sqrt[col]

            # Step 4-5: Start propagating messages.
            return self.propagate(edge_index_self_loop, x=x, norm=norm, edge_weight_self_loop=edge_weight_self_loop)
        def message(self, x_i, x_j, norm, edge_weight_self_loop):
            sign=torch.sign(edge_weight_self_loop).view(-1,1)
            x_j = (sign*x_j + x_i*(1 - sign))
            
            # Step 4: Normalize node features.
            return (norm.view(-1, 1) * x_j)
        

def feat_calculation(edge_index, feature):
    
    feat_diff = feature[edge_index[0]]-feature[edge_index[1]]
    feat_dist = torch.norm(feature[edge_index[0]]-feature[edge_index[1]],dim=1)
    
    return feat_diff, feat_dist
 

class GCNConv_nonlinear(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv_nonlinear, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = torch.nn.Linear(in_channels, out_channels)
            self.scalar1 = torch.nn.Linear(1, 1, bias=False)
            self.in_channels=in_channels

        def forward(self, x, edge_index, edge_weight):
            
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index_self_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight_self_loop = torch.cat((edge_weight, torch.ones(x.size(0))), dim=0)

            feat_diff, feat_dist =  feat_calculation(edge_index, x)
            zero_vector = torch.zeros(self.in_channels)
            feat_diff = torch.cat((feat_diff, zero_vector.repeat(x.size(0),1)))
            feat_dist = torch.cat((feat_dist, torch.zeros(x.size(0))))
            
            # Step 3: Compute normalization.
            row, col = edge_index_self_loop
            deg = scatter_add(np.abs(edge_weight_self_loop), col, dim=0, dim_size=x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * np.abs(edge_weight_self_loop) * deg_inv_sqrt[col]

            # Step 4-5: Start propagating messages.
            return self.propagate(edge_index_self_loop, x=x, norm=norm, edge_weight_self_loop=edge_weight_self_loop, edge_index_self_loop=edge_index_self_loop, feat_diff=feat_diff, feat_dist=feat_dist)

        def message(self, x_i, x_j, norm, edge_weight_self_loop, edge_index_self_loop, feat_diff, feat_dist):
        
            #find the position where feat_dist!=0 and feat_dist=0
            x_j1=torch.zeros(x_j.shape)
            x_j2=torch.zeros(x_j.shape)
            diff_location=np.where(feat_dist!=0)
            #reshape diff_location
            diff_loc = np.array(np.repeat(diff_location[0], self.in_channels))
            diff_loc = np.array((diff_loc, np.tile(np.arange(self.in_channels), len(diff_location[0]))))

            same_location=np.where(feat_dist==0)
            #reshape same_location
            same_loc = np.repeat(same_location[0], self.in_channels)
            same_loc = np.array((same_loc, np.tile(np.arange(self.in_channels), len(same_location[0]))))
   
            sign=torch.sign(edge_weight_self_loop).view(-1,1)
            
            x_j1[diff_loc]=x_i[diff_loc]+ sign[diff_loc[0]].view(-1)*(feat_diff[diff_loc]/feat_dist[diff_loc[0]])*torch.pow(feat_dist[diff_loc[0]]+(1-sign[diff_loc[0]].view(-1))/2,sign[diff_loc[0]].view(-1))
    
            #if x_i=x_j: it can be self-loop connections, and different nodes having same feature values
            x_j2[same_loc]=x_j[same_loc]
            
            #final x_j will be the sum of x_j1 and x_j2
            x_j=x_j1+x_j2
            
            #we multiply x_j with the trainable weights which correspond to hidden nodes
            x_j = self.lin(x_j)

            return norm.view(-1, 1) * x_j
        
class GCNConv_diff_split(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv_diff_split, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index_self_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight_self_loop = torch.cat((edge_weight, torch.ones(x.size(0))), dim=0)
            
            x = self.lin(x)
            pos_idx=torch.where(edge_weight_self_loop>0)[0]
            neg_idx=torch.where(edge_weight_self_loop<0)[0]
            # Step 3: Compute normalization.
            row_pos, col_pos = edge_index_self_loop[:, pos_idx]
            deg_pos = scatter_add(torch.abs(edge_weight_self_loop[pos_idx]), col_pos, dim=0, dim_size=x.size(0))
            deg_inv_sqrt_pos = deg_pos.pow(-1)
            norm_pos = deg_inv_sqrt_pos[col_pos] * torch.abs(edge_weight_self_loop[pos_idx])
            
            row_neg, col_neg = edge_index_self_loop[:, neg_idx]
            deg_neg = scatter_add(torch.abs(edge_weight_self_loop[neg_idx]), col_neg, dim=0, dim_size=x.size(0))
            deg_inv_sqrt_neg = deg_neg.pow(-1)
            norm_neg = deg_inv_sqrt_neg[col_neg] * torch.abs(edge_weight_self_loop[neg_idx])
            
            row, col = edge_index_self_loop
            deg = scatter_add(torch.abs(edge_weight_self_loop), col, dim=0, dim_size=x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * torch.abs(edge_weight_self_loop) * deg_inv_sqrt[col]
            norm[pos_idx]=norm_pos
            norm[neg_idx]=norm_neg

            # Step 4-5: Start propagating messages.
            return self.propagate(edge_index_self_loop, x=x, norm=norm, edge_weight_self_loop=edge_weight_self_loop)
        def message(self, x_i, x_j, norm, edge_weight_self_loop):
            sign=torch.sign(edge_weight_self_loop).view(-1,1)
            x_j = (x_i + sign*(x_j-x_i))
            
            # Step 4: Normalize node features.
            return (norm.view(-1, 1) * x_j)

class GCNConv_nonlinear_split(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNConv_nonlinear_split, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index_self_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_weight_self_loop = torch.cat((edge_weight, torch.ones(x.size(0))), dim=0)

            feat_diff, feat_dist =  feat_calculation(edge_index, x)
            zero_vector = torch.zeros(x.size(1))
            feat_diff = torch.cat((feat_diff, zero_vector.repeat(x.size(0),1)))
            feat_dist = torch.cat((feat_dist, torch.zeros(x.size(0))))
            
            # Step 3: Compute normalization.
            pos_idx=torch.where(edge_weight_self_loop>0)[0]
            neg_idx=torch.where(edge_weight_self_loop<0)[0]
            # Step 3: Compute normalization.
            row_pos, col_pos = edge_index_self_loop[:, pos_idx]
            deg_pos = scatter_add(torch.abs(edge_weight_self_loop[pos_idx]), col_pos, dim=0, dim_size=x.size(0))
            deg_inv_sqrt_pos = deg_pos.pow(-1)
            norm_pos = deg_inv_sqrt_pos[col_pos] * torch.abs(edge_weight_self_loop[pos_idx])
            
            row_neg, col_neg = edge_index_self_loop[:, neg_idx]
            deg_neg = scatter_add(torch.abs(edge_weight_self_loop[neg_idx]), col_neg, dim=0, dim_size=x.size(0))
            deg_inv_sqrt_neg = deg_neg.pow(-1)
            norm_neg = deg_inv_sqrt_neg[col_neg] * torch.abs(edge_weight_self_loop[neg_idx])
            
            row, col = edge_index_self_loop
            deg = scatter_add(torch.abs(edge_weight_self_loop), col, dim=0, dim_size=x.size(0))
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * torch.abs(edge_weight_self_loop) * deg_inv_sqrt[col]
            norm[pos_idx]=norm_pos
            norm[neg_idx]=norm_neg

            # Step 4-5: Start propagating messages.
            return self.propagate(edge_index_self_loop, x=x, norm=norm, edge_weight_self_loop=edge_weight_self_loop, edge_index_self_loop=edge_index_self_loop, feat_diff=feat_diff, feat_dist=feat_dist)

        def message(self, x, x_i, x_j, norm, edge_weight_self_loop, edge_index_self_loop, feat_diff, feat_dist):
        
            #find the position where feat_dist!=0 and feat_dist=0
            x_j1=torch.zeros(x_j.shape)
            x_j2=torch.zeros(x_j.shape)
            diff_location=np.where(feat_dist!=0)
            #reshape diff_location
            diff_loc = np.array(np.repeat(diff_location[0], x.size(1)))
            diff_loc = np.array((diff_loc, np.tile(np.arange(x.size(1)), len(diff_location[0]))))

            same_location=np.where(feat_dist==0)
            #reshape same_location
            same_loc = np.repeat(same_location[0], x.size(1))
            same_loc = np.array((same_loc, np.tile(np.arange(x.size(1)), len(same_location[0]))))
   
            sign=torch.sign(edge_weight_self_loop).view(-1,1)
            
            x_j1[diff_loc]=x_i[diff_loc]+ sign[diff_loc[0]].view(-1)*(feat_diff[diff_loc]/feat_dist[diff_loc[0]])*torch.pow(feat_dist[diff_loc[0]]+(1-sign[diff_loc[0]].view(-1))/2,sign[diff_loc[0]].view(-1))
    
            #if x_i=x_j: it can be self-loop connections, and different nodes having same feature values
            x_j2[same_loc]=x_j[same_loc]
            
            #final x_j will be the sum of x_j1 and x_j2
            x_j=x_j1+x_j2
            
            #we multiply x_j with the trainable weights which correspond to hidden nodes
            x_j = self.lin(x_j)

            return norm.view(-1, 1) * x_j
