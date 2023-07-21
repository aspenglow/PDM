import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import os
import numpy as np
import time

import sys
sys.path.append(os.getcwd())
from utils.position_encoding import LapEncoding
from utils.utils_latentgnn import edge_list_to_tensor, graph_to_edge_list
from latentgnn.latentgnn_v1 import LatentGNN
from utils.dataset import LayoutDataset
from utils.utils_neulay import energy 


def load_data(experiment_root_dir, graph_root_dir, layout_root_dir, dataset_ratio, re_split_dataset=False):
    dataset = LayoutDataset(graph_root_dir=graph_root_dir, layout_root_dir=layout_root_dir)
    train_size = int(len(dataset) * dataset_ratio[0])
    validate_size = int(len(dataset) * dataset_ratio[1])
    test_size = len(dataset) - validate_size - train_size
    print(graph_root_dir, layout_root_dir)
    print("data size:", len(dataset))
    print("train:", train_size, "val:", validate_size, "test:", test_size)
    
    if re_split_dataset:
        train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size, test_size])
        torch.save(train_dataset, os.path.join(graph_root_dir, "train_set.pkl"))
        torch.save(validate_dataset, os.path.join(graph_root_dir, "validate_set.pkl"))
        torch.save(test_dataset, os.path.join(graph_root_dir, "test_set.pkl"))
    else:
        try:
            train_dataset = torch.load(os.path.join(graph_root_dir, "train_set.pkl"))
            validate_dataset = torch.load(os.path.join(graph_root_dir, "validate_set.pkl"))
            test_dataset = torch.load(os.path.join(graph_root_dir, "test_set.pkl"))
        except:
            train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size, test_size])
            torch.save(train_dataset, os.path.join(graph_root_dir, "train_set.pkl"))
            torch.save(validate_dataset, os.path.join(graph_root_dir, "validate_set.pkl"))
            torch.save(test_dataset, os.path.join(graph_root_dir, "test_set.pkl"))

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("train set:", len(train_loader), " val set:", len(validate_loader), " test set:", len(test_loader))
    return data_loader, train_loader, validate_loader, test_loader

def one_norm_distance(layout, prediction_layout, variance=5e-5) -> torch.Tensor:
    neg_log_p = (torch.abs(layout - prediction_layout) / (2 * variance))
    N = prediction_layout.size(1)
    return neg_log_p.sum() / (N * N)

def two_norm_distance(layout, prediction_layout, variance=5e-5) -> torch.Tensor:
    neg_log_p = (torch.abs(layout - prediction_layout) ** 2 / (2 * variance))
    N = prediction_layout.size(1)
    return neg_log_p.sum() / (N * N)

def energy_loss(predict_energy, truth_energy):
    return torch.abs(predict_energy - truth_energy) / truth_energy 


def write_log(log_path, string):
    with open(log_path,'a') as f:
        f.write(string)