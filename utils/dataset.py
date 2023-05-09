import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import os
import numpy as np
import sys
sys.path.append(os.getcwd())

from latentgnn.position_encoding import LapEncoding
from utils.utils_latentgnn import graph_to_edge_list, edge_list_to_tensor


class LayoutDataset(Dataset):
    def __init__(self, graph_root_dir="graphs", layout_root_dir="layouts", encoding_dim=20):
        super().__init__()
        graph_dirs = os.listdir(graph_root_dir)
        layout_dirs = os.listdir(layout_root_dir)
        self.encoding_dim = encoding_dim
        self.graph_dirs = []
        self.layout_dirs = []
        
        for path in graph_dirs:
            if path not in layout_dirs:
                continue
            self.graph_dirs.append(os.path.join(graph_root_dir, path))
            self.layout_dirs.append(os.path.join(layout_root_dir, path))
    
    def __getitem__(self, index):
        try:
            dir_index = 0
            for graph_path in self.graph_dirs:
                num_graphs = len(os.listdir(graph_path))
                if index < num_graphs:
                    break
                index -= num_graphs
                dir_index += 1
            
            graph_name = os.listdir(self.graph_dirs[dir_index])[index]
            graph_name = graph_name.split(".")[0]
            graph_path = os.path.join(self.graph_dirs[dir_index], graph_name+".pkl")
            layout_path = os.path.join(self.layout_dirs[dir_index], graph_name+".pt")
            graph = nx.read_gpickle(graph_path)
            lapEncoding = LapEncoding(dim=self.encoding_dim)
            graph.edge_index = edge_list_to_tensor(graph_to_edge_list(graph))
            encoding = lapEncoding.compute_pe(graph)
            Adj = nx.to_numpy_matrix(graph)
            N = len(Adj)
            Adj =  Adj + np.eye(N)  #Laplacian mtx
            Deg = np.diag(np.array(1/np.sqrt(Adj.sum(0)))[0,:]) # degree mtx
            Adj_norm = np.dot(Deg, np.dot(Adj, Deg))
            Adj_norm = torch.from_numpy(Adj_norm)
            Adj_norm = Adj_norm.float()
            
            layout = torch.load(layout_path)
            return Adj_norm, encoding, layout, graph_name
        except EOFError:
            raise EOFError(str(index)+" "+graph_path+" "+layout_path)
    
    def __len__(self):
        return sum(len(os.listdir(path)) for path in self.graph_dirs)

