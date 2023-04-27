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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
tt = tictoc()

# log path
log_path = "neulay/log_neulay.txt"

# import a graph
data_dir = "graphs/erdos_renyi"
files = os.listdir(data_dir)[:10]
f = files[0]
G = nx.read_gpickle(os.path.join(data_dir, f))
A = nx.to_numpy_matrix(G)
N = len(A)

A = sp.coo_matrix(A)

adjacency_list = (torch.LongTensor(sp.triu(A, k=1).row), torch.LongTensor(sp.triu(A, k=1).col))
#adjacency_list = torch.where(torch.triu(torch.tensor(A)))

#propagation rule

#Adj =  A + np.eye(N)  #Laplacian mtx
#deg = np.diag(np.array(1/np.sqrt(Adj.sum(0)))[0,:]) # degree mtx
#DAD = np.dot(deg, np.dot(Adj, deg))
A_norm = A + sp.eye(N)
D_norm = sp.diags((1/np.sqrt(A_norm.sum(0))).tolist()[0])
D_norm = D_norm.tocsr()
DAD = D_norm.dot(A_norm.dot(D_norm))
DAD = DAD.tocoo()

DAD = sparse_mx_to_torch_sparse_tensor(DAD)



    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, adj_mx):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.adj_mx = adj_mx
        self.output_dim = output_dim
        #self.dense = nn.Linear(input_dim, output_dim,bias=False)
        self.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.input_dim, self.output_dim), gain= N**(1/dim))).to(device) 
        #torch.nn.Parameter(torch.rand(self.input_dim, self.output_dim)) #
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(self.adj_mx, support)
        
        return output
        
        
        return x
    
class LayoutNet(nn.Module):
    def __init__(self, num_nodes, output_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3,adj_mtx):
        super(LayoutNet, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        self.adj_mtx = adj_mtx
        
        #self.dense1 = nn.Linear(self.num_nodes, self.hidden_dim_1, bias= False)
        self.weight1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.num_nodes, self.hidden_dim_1), gain= N**(1/dim))).to(device)
        #torch.nn.Parameter(torch.rand(self.num_nodes, self.hidden_dim_1))#
        self.GCN1 = GCN(self.hidden_dim_1, self.hidden_dim_2,self.adj_mtx.float()).to(device)
        
        #self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh1 = nn.Tanh().to(device)
        #self.sigmoid = nn.Sigmoid()
        
        self.GCN2 = GCN(self.hidden_dim_2, self.hidden_dim_3, self.adj_mtx.float()).to(device)
       
        self.weight2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor((self.hidden_dim_1 + self.hidden_dim_2 + self.hidden_dim_3), self.output_dim), gain= N**(1/dim))).to(device)
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
    

#input, output dimensions
dim = 3 
#x = torch.eye(N) #.to(device)
x = sp.eye(N)
x = x.tocoo()
x = sparse_mx_to_torch_sparse_tensor(x)

# model
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, gain= N**(1/dim))

#loss function
radius = .4
magnitude = 100*N**(1/3)*radius
k = 1

def custom_loss(output, Dist):
    X = output
    X1 = X[adjacency_list[0]] 
    X2 = X[adjacency_list[1]] 
        
    V_el = (k/2)*torch.sum( torch.sum((X1-X2)**2, axis = -1))
    V_nn = magnitude * torch.sum(torch.exp(-Dist/4/(radius**2)))
   
    return V_el + V_nn
    
    
#energy

def energy(output):    
    X = output

    X1 = X[adjacency_list[0]] 
    X2 = X[adjacency_list[1]] 
        
    V_el = (k/2)*torch.sum( torch.sum((X1-X2)**2, axis = -1))
    r = X[...,np.newaxis,:] - X[...,np.newaxis,:,:]
    r2_len = torch.sum(r**2, axis = -1)
    V_nn = magnitude * torch.sum(torch.exp(-r2_len /4/(radius**2) ) ) 
    return V_el + V_nn

#stopping
stop_delta_ratio = 1e-3    
    
energy_hist = []
lowest_energy = float('inf') 
best_time_hist = []
time_hist = []
hist = []
output_ = []


for i in range(10):
    net = LayoutNet(num_nodes=N, output_dim=dim, hidden_dim_1=100, hidden_dim_2=100, hidden_dim_3=3, adj_mtx= DAD)
    net.to(device)
    
    net.apply(init_weights)
    print(net.GCN1)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    # criterion = custom_loss
    criterion = energy
    
    loss_history = [] 
    
    valid_losses_gcn = []
    valid_losses_lin = []
    
    
    patience = 10
    r = list(np.zeros(patience))
  
    tt.tic()
        
    for epoch in tqdm(range(40000), leave=False): 
        inp = x.to(device)
        print(next(net.parameters()).device, inp.device)
        optimizer.zero_grad()
        outputs = net(inp)
        
        # if epoch%5 ==0:
        #     pairs = c_kdtree(outputs, 4)
        # Dist = Distances_kdtree(outputs, pairs)
        # loss = criterion(outputs, Dist)
        loss = criterion(outputs)

        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_history.append(loss.item())
    
        
        r.append(loss.item())
        r.pop(0)
        
        scheduler.step()
        
        
        # if (difference(r)) < .0001*np.sqrt(N):
        if early_stopping(loss_history,stop_delta_ratio=stop_delta_ratio):
            time_hist += [tt.toc()]
            break
        
        time_hist += [tt.toc()]      
    
    write_log(log_path, 'Finished training gcn: ' +  str(epoch) + ' time: ' + str(tt.toc()) + ' energy: ' + str(loss) + "\n")
    
    w = torch.nn.Parameter(outputs.detach())
    net1 = LayoutLinear(w)
    optimizer1 = torch.optim.RMSprop(net1.parameters(), lr=0.01)
        
    for epoch1 in tqdm(range(epoch, 60000), leave=False): #60k
        inp = x.to(device)
    
        optimizer1.zero_grad()
        outputs1 = net1(inp)
        
        # if epoch1%5 ==0:
        #     pairs = c_kdtree(outputs1, 4)
        # Dist = Distances_kdtree(outputs1, pairs) 
        # loss1 = criterion(outputs1, Dist)
        loss1 = criterion(outputs1)

        loss1.backward(retain_graph=True)
        optimizer1.step()
        
        loss_history.append(loss1.item())
        
        r.append(loss1.item())
        r.pop(0)
        
        scheduler.step()
        
        # if (difference(r)) < 1e-8*np.sqrt(N):
        if early_stopping(loss_history,stop_delta_ratio=0.1*stop_delta_ratio):
            time_hist += [tt.toc()]           
            break
   
        time_hist += [tt.toc()]
    
    
    hist += [loss_history]
    energy_hist += [energy(outputs1).detach().numpy()]
    write_log(log_path, 'Finished training' + str(i) + ' epoch: ' + str(epoch1) + ' time: ' + str(tt.toc()) + ' energy: ' + str(energy_hist[-1]) + "\n")

    if energy_hist[-1] < lowest_energy:
        write_log(log_path, "Better result with energy: " + str(energy_hist[-1]) + "\n")
        lowest_energy = energy_hist[-1]
        best_outputs = outputs1
    
d = pd.DataFrame(energy_hist)
d.to_csv('./neulay/internet_energy_neulay.csv', header=True,index=False)

d = pd.DataFrame(time_hist)
d.to_csv('./neulay/internet_time_neulay.csv', header=True,index=False)

d = pd.DataFrame(hist)
d.to_csv('./neulay/internet_loss_neulay.csv', header=True,index=False)

d = pd.DataFrame(outputs1.detach().numpy())
d.to_csv('./neulay/internet_output_neulay.csv', header=True,index=False)





