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
from utils.utils_neulay import *
from neulay.neulay_model import *

import argparse
parser = argparse.ArgumentParser(
    'Force-Directed Layout algorithm to calculate final layout given a graph.')
parser.add_argument('--data-dir', type=str, default="graphs_sbm/sbm",
                    help='Path to load graphs.')
parser.add_argument('--graphs-num', type=int, default=-1,
                    help='Number of graph to train. -1 means load all of graphs in data-dir.')
parser.add_argument('--graphs-start-at', type=int, default=1,
                    help='From which picture to start training.')
parser.add_argument('--layout-dir', type=str, default="layouts_sbm/sbm",
                    help='Path to save trained graph layouts.')
parser.add_argument('--layout_dim', type=int, default=3,
                    help='Dimension of graph layout.')
parser.add_argument('--log-path', type=str, default="neulay/log_neulay.txt",
                    help='Path to save training log.')
parser.add_argument('--csv-dir', type=str, default=None,
                    help='Path to save results of loss and time.')
parser.add_argument('--train-num', type=int, default=3,
                    help='Number of training per graph, then save the layout with lowest loss.')
parser.add_argument('--stop-delta-ratio', type=float, default=1e-4,
                    help='A parameter to early stop the training.')
parser.add_argument('--only-cpu', action="store_true",
                    help='Use gpu for training.')
args = parser.parse_args()

if args.only_cpu:
    device = 'cpu'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
# measure time
tt = tictoc()

# log path
log_path = args.log_path

# layout save dir
layout_dir = args.layout_dir 
if layout_dir is not None:
    os.makedirs(layout_dir, exist_ok=True)

# import graphs
data_dir = args.data_dir
graphs_start_at = args.graphs_start_at
graphs_num = args.graphs_num
files = os.listdir(data_dir)
if graphs_start_at > 1:
    files = files[graphs_start_at-1:]
if graphs_num > -1:
    files = files[:graphs_num]
    
for f in tqdm(files, leave=False):
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

    #input, output dimensions
    layout_dim = args.layout_dim
    #x = torch.eye(N) #.to(device)
    x = sp.eye(N)
    x = x.tocoo()
    x = sparse_mx_to_torch_sparse_tensor(x)

    # model
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain= N**(1/layout_dim))

    #loss function
    radius = .4
    magnitude = 100*N**(1/3)*radius
    k = 1



    #stopping
    stop_delta_ratio = args.stop_delta_ratio
        
    energy_hist = []
    lowest_energy = float('inf') 
    best_time_hist = []
    time_hist = []
    hist = []
    output_ = []


    for i in tqdm(range(args.train_num), leave=False):
        net = LayoutNet(num_nodes=N, output_dim=layout_dim, hidden_dim_1=100, hidden_dim_2=100, hidden_dim_3=3, adj_mtx= DAD)
        net.to(device)
        
        net.apply(init_weights)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
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
            optimizer.zero_grad()
            outputs = net(inp)
            
            # if epoch%5 ==0:
            #     pairs = c_kdtree(outputs, 4)
            # Dist = Distances_kdtree(outputs, pairs)
            # loss = criterion(outputs, Dist)
            loss = criterion(outputs, adjacency_list, N)

            loss.backward(retain_graph=True)
            optimizer.step()
            
            loss_history.append(loss.item())
        
            
            r.append(loss.item())
            r.pop(0)
            
            scheduler.step()
            
            
            # if (difference(r)) < .0001*np.sqrt(N):
            if early_stopping(loss_history,stop_delta_ratio=10*stop_delta_ratio):
                time_hist += [tt.toc()]
                break
            
            time_hist += [tt.toc()]      
        
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
            loss1 = criterion(outputs1, adjacency_list, N)

            loss1.backward(retain_graph=True)
            optimizer1.step()
            
            loss_history.append(loss1.item())
            
            r.append(loss1.item())
            r.pop(0)
            
            scheduler.step()
            
            # if (difference(r)) < 1e-8*np.sqrt(N):
            if early_stopping(loss_history,stop_delta_ratio=stop_delta_ratio):
                time_hist += [tt.toc()]           
                break
    
            time_hist += [tt.toc()]
        
        
        hist += [loss_history]
        energy_hist += [loss1.item()]
        write_log(log_path, 'Graph: ' + f + ' Nodes: ' + str(N) + ' Finished training ' + str(i) + ' epoch: ' + str(epoch1) + \
                ' time: ' + str(tt.toc()) + ' energy: ' + str(energy_hist[-1]) + "\n")

        if energy_hist[-1] < lowest_energy:
            write_log(log_path, "Better result with energy: " + str(energy_hist[-1]) + "\n")
            lowest_energy = energy_hist[-1]
            best_outputs = outputs1
    
    # Remove mean, make center of nodes as zero point.
    best_outputs = best_outputs - torch.mean(best_outputs, axis=0)
    
    write_log(log_path, "\n")    
    graph_name = f.split('.')[0]
    if layout_dir is not None:
        torch.save(best_outputs, os.path.join(layout_dir, graph_name + '.pt'))
            
    if args.csv_dir is not None:    
        d = pd.DataFrame(energy_hist)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_energy_neulay.csv'), header=True,index=False)

        d = pd.DataFrame(time_hist)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_time_neulay.csv'), header=True,index=False)

        d = pd.DataFrame(hist)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_loss_neulay.csv'), header=True,index=False)

        d = pd.DataFrame(best_outputs.detach().cpu().numpy())
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_output_neulay.csv'), header=True,index=False)





