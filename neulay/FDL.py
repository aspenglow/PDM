#import packages
import networkx as nx
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial
from scipy.spatial import cKDTree
import scipy.sparse as sp
import time
import os
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from neulay.neulay_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# measure time
tt = tictoc()

# log path
log_path = "neulay/log_fdl.txt"

# introduce a graph
data_dir = "graphs/erdos_renyi"
files = os.listdir(data_dir)[:10]
f = files[0]
G = nx.read_gpickle(os.path.join(data_dir, f))
A = nx.to_numpy_matrix(G)
N = len(A)
A = sp.coo_matrix(A) #sparsification of A

adjacency_list = (torch.LongTensor(sp.triu(A, k=1).row), torch.LongTensor(sp.triu(A, k=1).col))
# adjacency_list = torch.where(torch.triu(torch.tensor(A)))
    
#FDL model
class LayoutLinear(nn.Module):
    def __init__(self, weight):
        super(LayoutLinear, self).__init__()
        self.weight = weight
        
    def forward(self, inp):
        x = torch.spmm(inp, self.weight)
        return x
    
#input, output dimensions
dim = 3 
# x = torch.eye(N) #.to(device)

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
stop_delta_ratio = 1e-4

#optimizer    
    
energy_hist_lin = []
lowest_energy = float('inf') 
best_time_hist = []
time_hist_lin = []
hist = []

for i in range(10):
    net = nn.Linear(N, dim, bias=False)
    net.to(device)
    net.apply(init_weights)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)
    # criterion = custom_loss
    criterion = energy
    
    loss_history_lin= [] 
    time_hist = []
    
    patience = 5
    r = list(np.zeros(patience))
    
    tt.tic()
    
    for epoch in tqdm(range(50000), leave=False):  
        inp = x.to(device)
    
        optimizer.zero_grad()
        
        outputsLin = net(inp)
        
        # if epoch%5 ==0:
        #     pairs = c_kdtree(outputsLin, 4)
        
        # Dist = Distances_kdtree(outputsLin, pairs)
        # loss = criterion(outputsLin, Dist)
        loss = criterion(outputsLin)
    
        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_history_lin.append(loss.item())

        r.append(loss.item())
        r.pop(0)
        
        scheduler.step()
        
        time_hist.append(tt.toc())
        
        # if (difference(r)) < 1e-8*np.sqrt(N):
        if early_stopping(loss_history_lin,stop_delta_ratio=stop_delta_ratio):
            time_hist_lin += [time_hist]
            break
    
    hist += [loss_history_lin]
    energy_hist_lin += [loss.detach().cpu().numpy()]
    # energy_hist_lin += [energy(outputsLin).detach().cpu().numpy()]
    # print(loss, " ", energy(outputsLin).detach().cpu().numpy())
    write_log(log_path, 'Finished training ' + str(i) + ' epoch: ' + str(epoch) + ' time: ' + str(tt.toc()) + ' energy: ' + str(energy_hist_lin[-1]) + "\n")
    print('Finished training '+ str(i) + ' epoch: ' + str(epoch) + ' time: ' + str(tt.toc()) + ' energy: ' + str(energy_hist_lin[-1]))
    
    if energy_hist_lin[-1] < lowest_energy:
        write_log(log_path, "Better result with energy: " + str(energy_hist_lin[-1]) + "\n")
        lowest_energy = energy_hist_lin[-1]
        best_outputs = outputsLin
    
d = pd.DataFrame(energy_hist_lin)
d.to_csv('./neulay/internet_energy_fdl.csv', header=True,index=False)

d = pd.DataFrame(time_hist_lin)
d.to_csv('./neulay/internet_time_fdl.csv', header=True,index=False)

d = pd.DataFrame(hist)
d.to_csv('./neulay/internet_loss_fdl.csv', header=True,index=False)

d = pd.DataFrame(best_outputs.detach().cpu().numpy())
d.to_csv('./neulay/internet_output_fdl.csv', header=True,index=False)

