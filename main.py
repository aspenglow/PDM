import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import orthogonal_procrustes
import networkx as nx
import os
import numpy as np
import time
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from latentgnn.position_encoding import LapEncoding
from latentgnn.utils_latentgnn import edge_list_to_tensor, graph_to_edge_list
from latentgnn.latentgnn_v1 import LatentGNN
from dataset import LayoutDataset
from utils import load_data, one_norm_distance, write_log
from neulay.utils_neulay import energy 


import argparse

parser = argparse.ArgumentParser(
    'Train LatentGNN model to predict final layout from different graphs.')
parser.add_argument('--graph-root-dir', type=str, default="graphs/",
                    help='Root path to load graphs.')
parser.add_argument('--layout-root-dir', type=str, default="layouts/",
                    help='Root path to load final layouts.')
parser.add_argument('--train-set-ratio', type=float, default=0.7,
                    help='Ratio of train set.')
parser.add_argument('--val-set-ratio', type=float, default=0.2,
                    help='Ratio of validation set.')
parser.add_argument('--position-encoding-dim', type=int, default=100,
                    help='Dim of graph position encoding as input feature.')
parser.add_argument('--latent-dims', type=list, default=[50,50,50],
                    help='Number of nodes for each latent graph layer.')
parser.add_argument('--channel-multi', type=int, default=20,
                    help='How many multiple channels in latent space compare with visible space.')
parser.add_argument('--log-dir', type=str, default="log",
                    help='Dir to save train validation and test logs.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Numer of epoch to train.')
parser.add_argument('--use-gpu', type=bool, default=True,
                    help='Use gpu to  training.')

args = parser.parse_args()

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

graph_root_dir = args.graph_root_dir
layout_root_dir = args.layout_root_dir

train_set_ratio = args.train_set_ratio
val_set_ratio = args.val_set_ratio
assert(train_set_ratio + val_set_ratio < 1.0)
test_set_ratio = 1.0 - train_set_ratio - val_set_ratio
dataset_ratio = [train_set_ratio, val_set_ratio, test_set_ratio]

encoding_dim = args.position_encoding_dim
latent_dims = args.latent_dims
channel_multi = args.channel_multi

epochs = args.epochs
log_dir = args.log_dir


train_loader, validation_loader, test_loader = load_data(graph_root_dir, layout_root_dir, dataset_ratio, encoding_dim)

model = LatentGNN(in_features=encoding_dim, out_features=3, latent_dims=latent_dims, channel_multi=channel_multi)  # 加载模型
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), 0.01) # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
criterion = one_norm_distance

def train(epoch):
    t = time.time()
    model.train()
    
    average_loss = 0.
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(train_loader), desc="train", total=len(train_loader), ncols=100, leave=False):
        optimizer.zero_grad()
        Adj = Adj.to(device)
        feature = feature.to(device)
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, len(Adj[0]))
        truth_energy = energy(layout[0], adjacency_list, len(Adj[0]))
        # print("predict: ", predict_energy, " truth: ", truth_energy)
        R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
        R = torch.from_numpy(R).unsqueeze(0)
        # layout = torch.bmm(layout, R)
        # loss = one_norm_distance(predict_layout, layout)
        loss = torch.abs(predict_energy - truth_energy) / (layout.size(0) * layout.size(1))
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        
        average_loss += loss.item() / len(train_loader)
        
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+str(graph_name)+" N_nodes: "+str(len(Adj[0]))+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "train_log.txt"), log_string + "\n")
        
    write_log(os.path.join(log_dir, "train_log.txt"), "average loss: "+str(average_loss)+"\n\n")
    
    
    model.eval()
    average_loss = 0.
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(validation_loader), desc="val", total=len(validation_loader), ncols=100, leave=False):
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, len(Adj[0]))
        truth_energy = energy(layout[0], adjacency_list, len(Adj[0]))
        # print("predict: ", predict_energy, " truth: ", truth_energy)
        R = orthogonal_procrustes(layout[0].detach().numpy(), predict_layout[0].detach().numpy())[0]
        R = torch.from_numpy(R).unsqueeze(0)
        # layout = torch.bmm(layout, R)
        # loss = one_norm_distance(predict_layout, layout)
        loss = torch.abs(predict_energy - truth_energy) / (layout.size(0) * layout.size(1))
        average_loss += loss.item() / len(validation_loader)
        
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(len(Adj[0]))+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "val_log.txt"), log_string + "\n")
    write_log(os.path.join(log_dir, "val_log.txt"), "average loss: "+str(average_loss)+"\n\n")
    
def test():
    model.eval()
    average_loss = 0.
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(test_loader), desc="test", total=len(test_loader), ncols=100, leave=False):
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, len(Adj[0]))
        truth_energy = energy(layout[0], adjacency_list, len(Adj[0]))
        # print("predict: ", predict_energy, " truth: ", truth_energy)
        R = orthogonal_procrustes(layout[0].detach().numpy(), predict_layout[0].detach().numpy())[0]
        R = torch.from_numpy(R).unsqueeze(0)
        # layout = torch.bmm(layout, R)
        # loss = one_norm_distance(predict_layout, layout)
        loss = torch.abs(predict_energy - truth_energy) / (layout.size(0) * layout.size(1))
        average_loss += loss.item() / len(test_loader)
        
        log_string = " graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(len(Adj[0]))+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "test_log.txt"), log_string + "\n")
        
    write_log(os.path.join(log_dir, "test_log.txt"), "average loss: "+str(average_loss)+"\n\n")

def main():
    os.makedirs(log_dir, exist_ok=True)
    for epoch in tqdm(range(epochs), desc="epoch", ncols=80, leave=False):
        train(epoch)
    test()
    
if __name__ == "__main__":
    print(len(os.listdir("layouts/erdos_renyi")), len(os.listdir("layouts/sbm")))
    main()