import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import os
import numpy as np
import time

from latentgnn.position_encoding import LapEncoding
from latentgnn.utils_latentgnn import edge_list_to_tensor, graph_to_edge_list
from latentgnn.latentgnn_v1 import LatentGNN
from dataset import LayoutDataset
from neulay.utils_neulay import energy 


def load_data(graph_root_dir, layout_root_dir, dataset_ratio, encoding_dim):
    dataset = LayoutDataset(graph_root_dir=graph_root_dir, layout_root_dir=layout_root_dir, encoding_dim=encoding_dim)
    train_size = int(len(dataset) * dataset_ratio[0])
    validate_size = int(len(dataset) * dataset_ratio[1])
    test_size = len(dataset) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size, test_size])

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("train set:", len(train_loader), " val set:", len(validate_loader), " test set:", len(test_loader))
    return train_loader, validate_loader, test_loader

def one_norm_distance(layout, prediction_layout, variance=5e-5) -> torch.Tensor:
    neg_log_p = (torch.abs(layout - prediction_layout) / (2 * variance))
    return neg_log_p.sum() / (prediction_layout.size(1) * prediction_layout.size(2))


def write_log(log_path, string):
    with open(log_path,'a') as f:
        f.write(string)