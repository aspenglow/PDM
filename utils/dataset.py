import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import os
import numpy as np
import sys
sys.path.append(os.getcwd())

from utils.position_encoding import LapEncoding
from utils.utils_latentgnn import graph_to_edge_list, edge_list_to_tensor


class LayoutDataset(Dataset):
    def __init__(self, graph_root_dir="graphs", layout_root_dir="layouts"):
        super().__init__()
        graph_dirs = os.listdir(graph_root_dir)
        layout_dirs = os.listdir(layout_root_dir)
        self.graph_dirs = []
        self.layout_dirs = []
        
        for path in graph_dirs:
            if os.path.isdir(os.path.join(graph_root_dir, path)):
                if path not in layout_dirs:
                    continue   
                self.graph_dirs.append(os.path.join(graph_root_dir, path))
                self.layout_dirs.append(os.path.join(layout_root_dir, path))
        print(graph_dirs, self.graph_dirs, layout_dirs, self.layout_dirs)
    
    def __getitem__(self, index):
        # return (normalized adjcency matrix of the graph, position encoding, layout of the graph, graph name)
        try:
            dir_index = 0
            for graph_path in self.graph_dirs:
                num_graphs = len(os.listdir(graph_path))
                if index < num_graphs:
                    break
                index -= num_graphs
                dir_index += 1
            
            graph_dir = self.graph_dirs[dir_index]
            graph_name = os.listdir(graph_dir)[index]
            graph_name = graph_name.split(".")[0]
            graph_path = os.path.join(self.graph_dirs[dir_index], graph_name+".pkl")
            layout_path = os.path.join(self.layout_dirs[dir_index], graph_name+".pt")
            graph = nx.read_gpickle(graph_path)
            
            Adj = nx.to_numpy_matrix(graph)
            N = len(Adj)
            Adj =  Adj + np.eye(N)  #Laplacian mtx
            Deg = np.diag(np.array(1/np.sqrt(Adj.sum(0)))[0,:]) # degree mtx
            Adj_norm = np.dot(Deg, np.dot(Adj, Deg))
            Adj_norm = torch.from_numpy(Adj_norm)
            Adj_norm = Adj_norm.float()
            
            layout = torch.load(layout_path)
            # remove mean
            layout = layout - torch.mean(layout, axis=0)
            return Adj_norm, layout, os.path.join(os.path.split(graph_dir)[1], graph_name)
        except EOFError:
            raise EOFError(str(index)+" "+graph_path+" "+layout_path)
    
    def __len__(self):
        return sum(len(os.listdir(path)) for path in self.graph_dirs)

