#import packages
import networkx as nx
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
import math

import sys
sys.path.append(os.getcwd())
from neulay.neulay_utils import *

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, adj_mx, N, device=None, layout_dim=3):
        super(GCN, self).__init__()
        if device==None:
            device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.adj_mx = adj_mx.to(device)
        self.output_dim = output_dim
        #self.dense = nn.Linear(input_dim, output_dim,bias=False)
        self.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.input_dim, self.output_dim), gain= N**(1/layout_dim)).to(self.device))
        #torch.nn.Parameter(torch.rand(self.input_dim, self.output_dim)) #
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(self.adj_mx, support)
        
        return output
    
class LayoutNet(nn.Module):
    def __init__(self, num_nodes, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, adj_mtx, N, device=None, layout_dim=3):
        super(LayoutNet, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        self.adj_mtx = adj_mtx
        if device==None:
            device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        
        #self.dense1 = nn.Linear(self.num_nodes, self.hidden_dim_1, bias= False)
        self.weight1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.num_nodes, self.hidden_dim_1), gain= N**(1/layout_dim)).to(self.device))
        #torch.nn.Parameter(torch.rand(self.num_nodes, self.hidden_dim_1))#
        self.GCN1 = GCN(self.hidden_dim_1, self.hidden_dim_2,self.adj_mtx.float(), N, device=device).to(self.device)
        
        #self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh1 = nn.Tanh().to(self.device)
        #self.sigmoid = nn.Sigmoid()
        
        self.GCN2 = GCN(self.hidden_dim_2, self.hidden_dim_3, self.adj_mtx.float(), N, device=device).to(self.device)
       
        self.weight2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor((self.hidden_dim_1 + self.hidden_dim_2 + self.hidden_dim_3), self.output_dim), gain= N**(1/layout_dim)).to(self.device))
        #torch.nn.Parameter(torch.rand((self.hidden_dim_1 + self.hidden_dim_2+ self.hidden_dim_3), self.output_dim)) #
       
        #self.dense2 = nn.Linear((self.hidden_dim_1 + self.hidden_dim_2+ self.hidden_dim_3), self.output_dim, bias= False)
       
 
    def forward(self, inp):
        x = torch.spmm(inp, self.weight1)
        #x = self.dense1(inp)

        gnn1 = self.GCN1(x)
        
        #gnn1 = self.leakyrelu(gnn1)
        gnn1 = self.tanh1(gnn1)
        #gnn1 = self.sigmoid(gnn1)
        
        gnn2 = self.GCN2(gnn1)
        
        
        output = torch.cat((x,gnn1,gnn2),1)
        
        #output = self.dense2(output)
        output = torch.spmm(output, self.weight2)
    
        
        return  output

class LayoutLinear(nn.Module):
    def __init__(self, weight):
        super(LayoutLinear, self).__init__()
        self.weight = weight
        
    def forward(self, inp):
        x = torch.spmm(inp, self.weight)
        
        return  x
    