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

import argparse
parser = argparse.ArgumentParser(
    'Force-Directed Layout algorithm to calculate final layout given a graph.')
parser.add_argument('--data-dir', type=str, default="graphs/erdos_renyi",
                    help='Path to load graphs.')
parser.add_argument('--graph-num', type=int, default=5,
                    help='Number of graph to train. -1 means load all of graphs in data-dir.')
parser.add_argument('--layout-dir', type=str, default="layouts/erdos_renyi",
                    help='Path to save trained graph layouts.')
parser.add_argument('--layout_dim', type=int, default=3,
                    help='Dimension of graph layout.')
parser.add_argument('--log-path', type=str, default="neulay/log_fdl.txt",
                    help='Path to save training log.')
parser.add_argument('--csv-dir', type=str, default=None,
                    help='Path to save results of loss and time.')
parser.add_argument('--train-num', type=int, default=5,
                    help='Number of training per graph, then save the layout with lowest loss.')
parser.add_argument('--stop-delta-ratio', type=int, default=1e-4,
                    help='A parameter to early stop the training.')
parser.add_argument('--use-gpu', type=bool, default=True,
                    help='Use gpu to  training.')
args = parser.parse_args()

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
    
# measure time
tt = tictoc()

# log path
log_path = args.log_path

# layout save dir
layout_dir = args.layout_dir 
os.makedirs(layout_dir, exist_ok=True)

# introduce graphs
data_dir = args.data_dir
graphs_num = args.graph_num
if graphs_num == -1:
    files = os.listdir(data_dir)
else:
    files = os.listdir(data_dir)[:graphs_num]

for f in tqdm(files, leave=False):
    G = nx.read_gpickle(os.path.join(data_dir, f))
    A = nx.to_numpy_matrix(G)
    N = len(A)
    A = sp.coo_matrix(A) #sparsification of A

    adjacency_list = (torch.LongTensor(sp.triu(A, k=1).row), torch.LongTensor(sp.triu(A, k=1).col))
    # adjacency_list = torch.where(torch.triu(torch.tensor(A)))
        
    #input, output dimensions
    layout_dim = args.layout_dim
    # x = torch.eye(N) #.to(device)

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
    stop_delta_ratio = 1e-3

    #optimizer    
        
    energy_hist_lin = []
    lowest_energy = float('inf') 
    best_time_hist = []
    time_hist_lin = []
    hist = []

    for i in tqdm(range(args.train_num), leave=False):
        net = nn.Linear(N, layout_dim, bias=False)
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
            loss = criterion(outputsLin, adjacency_list, N)
        
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
        # print('Finished training '+ str(i) + ' epoch: ' + str(epoch) + ' time: ' + str(tt.toc()) + ' energy: ' + str(energy_hist_lin[-1]))
        
        if energy_hist_lin[-1] < lowest_energy:
            write_log(log_path, "Better result with energy: " + str(energy_hist_lin[-1]) + "\n")
            lowest_energy = energy_hist_lin[-1]
            best_outputs = outputsLin
    
    graph_name = f.split('.')[0]
    torch.save(best_outputs, os.path.join(layout_dir, graph_name + '.pt'))
    
    if args.csv_dir is not None:    
        d = pd.DataFrame(energy_hist_lin)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_energy_fdl.csv'), header=True,index=False)

        d = pd.DataFrame(time_hist_lin)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_time_fdl.csv'), header=True,index=False)

        d = pd.DataFrame(hist)
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_loss_fdl.csv'), header=True,index=False)

        d = pd.DataFrame(best_outputs.detach().cpu().numpy())
        d.to_csv(os.path.join(args.csv_dir, graph_name + '_output_fdl.csv'), header=True,index=False)

