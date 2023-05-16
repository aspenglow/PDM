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
from utils.position_encoding import LapEncoding
from utils.utils_latentgnn import edge_list_to_tensor, graph_to_edge_list
from latentgnn.latentgnn_v1 import LatentGNN
from utils.dataset import LayoutDataset
from utils.utils import load_data, one_norm_distance, energy_loss, write_log
from utils.utils_neulay import energy 


import argparse

parser = argparse.ArgumentParser(
    'Train LatentGNN model to predict final layout from different graphs.')
parser.add_argument('--graph-root-dir', type=str, default="graphs_sbm/",
                    help='Root path to load graphs.')
parser.add_argument('--layout-root-dir', type=str, default="layouts_sbm/",
                    help='Root path to load final layouts.')
parser.add_argument('--output-root-dir', type=str, default="outputs_sbm/",
                    help='Root path to save predicted final layouts.')
parser.add_argument('--model-save-dir', type=str, default="model_sbm/",
                    help='Dir to save best model.')
parser.add_argument('--model-load-path', type=str, default=None,
                    help='Path to load trainer models. "None" means train a new model. ')
parser.add_argument('--log-dir', type=str, default="log_sbm",
                    help='Dir to save train validation and test logs.')
parser.add_argument('--train-set-ratio', type=float, default=0.7,
                    help='Ratio of train set.')
parser.add_argument('--val-set-ratio', type=float, default=0.2,
                    help='Ratio of validation set.')
parser.add_argument('--position-encoding-dim', type=int, default=50,
                    help='Dim of graph position encoding as input feature.')
parser.add_argument('--latent-dims', type=list, default=[2,2,2],
                    help='Number of nodes for each latent graph layer.')
parser.add_argument('--channel-multi', type=int, default=20,
                    help='How many multiple channels in latent space compare with visible space.')
parser.add_argument('--mode', type=str, default="asymmetric",
                    help='Which kind of latentGNN to use (symmetric/asymmetric).')
parser.add_argument('--loss', type=str, default="energy",
                    help='Which kind of loss function to use (energy/coordinate_diff).')
parser.add_argument('--epochs', type=int, default=100,
                    help='Numer of epoch to train.')
parser.add_argument('--gamma', type=float, default=0.97,
                    help='Parameter to decay the learning rate per epoch.')
parser.add_argument('--only-cpu', action="store_true",
                    help='Use gpu for training.')
parser.add_argument('--no-train', action="store_true",
                    help='If train model.')
parser.add_argument('--no-test', action="store_true",
                    help='If test model.')
parser.add_argument('--no-predict', action="store_true",
                    help='If use trained model to predict and save graph layout.')

args = parser.parse_args()

if args.only_cpu:
    device = 'cpu'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_root_dir = args.graph_root_dir
layout_root_dir = args.layout_root_dir
output_root_dir = args.output_root_dir
model_save_dir = args.model_save_dir
model_load_path = args.model_load_path

train_set_ratio = args.train_set_ratio
val_set_ratio = args.val_set_ratio
assert(train_set_ratio + val_set_ratio < 1.0)
test_set_ratio = 1.0 - train_set_ratio - val_set_ratio
dataset_ratio = [train_set_ratio, val_set_ratio, test_set_ratio]

encoding_dim = args.position_encoding_dim
latent_dims = args.latent_dims
channel_multi = args.channel_multi

epochs = args.epochs
gamma = args.gamma
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

no_train = args.no_train
no_test = args.no_test
no_predict = args.no_predict

# Load dataloader
data_loader, train_loader, validation_loader, test_loader = load_data(graph_root_dir, layout_root_dir, dataset_ratio, encoding_dim)

# Initialize models
model = LatentGNN(in_features=encoding_dim, out_features=3, latent_dims=latent_dims, channel_multi=channel_multi, mode=args.mode)  # 加载模型
if model_load_path is not None:
    model.load_state_dict(torch.load(model_load_path))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

def train(epoch, best_val_loss) -> float:
    # Training
    model.train()
    average_train_loss = 0.
    avg_train_pre_energy = 0.
    avg_train_tru_energy = 0.
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(train_loader), desc="train", total=len(train_loader), ncols=80, leave=False):
        optimizer.zero_grad()
        Adj = Adj.to(device)
        N = Adj.size(1)
        layout = layout.to(device)
        feature = feature.to(device)
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        # print("predict: ", predict_energy, " truth: ", truth_energy)
        
        if args.loss == "energy":
            loss = energy_loss(predict_energy, truth_energy)
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = one_norm_distance(predict_layout, layout)

        loss.backward()
        
        average_train_loss += loss.item() / len(train_loader)
        avg_train_pre_energy += predict_energy.item() / len(train_loader)
        avg_train_tru_energy += truth_energy.item() / len(train_loader)
        
        graph_name = graph_name[0]
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "train_log.txt"), log_string + "\n")
    
    log_string = "epoch "+str(epoch)+" average loss: "+str(average_train_loss)+" avg pre energy: "+str(avg_train_pre_energy)+" avg truth energy: "+str(avg_train_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
    write_log(os.path.join(log_dir, "train_log.txt"), log_string+"\n\n")
    write_log(os.path.join(log_dir, "train_log_summary.txt"), log_string+"\n")
    
    optimizer.step()
    scheduler.step()
    
    # Validation
    model.eval()
    average_val_loss = 0.
    avg_val_pre_energy = 0.
    avg_val_tru_energy = 0.
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(validation_loader), desc="val", total=len(validation_loader), ncols=80, leave=False):
        Adj = Adj.to(device)
        N = Adj.size(1)
        feature = feature.to(device)
        layout = layout.to(device)
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        if args.loss == "energy":
            loss = energy_loss(predict_energy, truth_energy)
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = one_norm_distance(predict_layout, layout)
        
        average_val_loss += loss.item() / len(validation_loader)
        avg_val_pre_energy += predict_energy.item() / len(validation_loader)
        avg_val_tru_energy += truth_energy.item() / len(validation_loader)
        
        graph_name = graph_name[0]
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "val_log.txt"), log_string + "\n")
    
    log_string = "epoch "+str(epoch)+" average loss: "+str(average_val_loss)+" avg pre energy: "+str(avg_val_pre_energy)+" avg truth energy: "+str(avg_val_tru_energy)
    write_log(os.path.join(log_dir, "val_log.txt"), log_string+"\n\n")
    write_log(os.path.join(log_dir, "val_log_summary.txt"), log_string+"\n")
    
    if average_val_loss < best_val_loss:
        write_log(os.path.join(log_dir, "val_log.txt"), "Better model! saving..."+"\n\n")
        write_log(os.path.join(log_dir, "val_log_summary.txt"), "Better model! saving..."+"\n\n")
        if model_save_dir is not None:
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, "model.pt"))
    return average_val_loss
   
    
def test(model_load_path: str=None) -> float:
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
    model.eval()
    
    average_test_loss = 0.
    avg_test_pre_energy = 0.
    avg_test_tru_energy = 0.
    
    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(test_loader), desc="test", total=len(test_loader), ncols=80):
        Adj = Adj.to(device)
        N = Adj.size(1)
        feature = feature.to(device)
        layout = layout.to(device)
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        if args.loss == "energy":
            loss = energy_loss(predict_energy, truth_energy)
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = one_norm_distance(predict_layout, layout)
        
        average_test_loss += loss.item() / len(test_loader)
        avg_test_pre_energy += predict_energy.item() / len(test_loader)
        avg_test_tru_energy += truth_energy.item() / len(test_loader)
        
        graph_name = graph_name[0]
        log_string = " graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "test_log.txt"), log_string + "\n")
    
    log_string = "average loss: "+str(average_test_loss)+" avg pre energy: "+str(avg_test_pre_energy)+" avg truth energy: "+str(avg_test_tru_energy)    
    write_log(os.path.join(log_dir, "test_log.txt"), log_string+"\n\n")
    return average_test_loss

def predict(model_load_path: str=None):
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
    model.eval()
    
    average_predict_loss = 0.
    avg_predict_pre_energy = 0.
    avg_predict_tru_energy = 0.
    
    dirs = os.listdir(layout_root_dir)
    for dir in dirs:
        os.makedirs(os.path.join(output_root_dir, dir), exist_ok=True)

    for batch_idx, (Adj, feature, layout, graph_name) in tqdm(enumerate(data_loader), desc="predict", total=len(data_loader), ncols=80):
        Adj = Adj.to(device)
        N = Adj.size(1)
        feature = feature.to(device)
        layout = layout.to(device)
        predict_layout = model(Adj, feature)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        if args.loss == "energy":
            loss = energy_loss(predict_energy, truth_energy)
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = one_norm_distance(predict_layout, layout)
        
        average_predict_loss += loss.item() / len(data_loader)
        avg_predict_pre_energy += predict_energy.item() / len(data_loader)
        avg_predict_tru_energy += truth_energy.item() / len(data_loader)
        
        graph_name: str = graph_name[0]
        if graph_name.startswith("er"):
            torch.save(predict_layout, os.path.join(output_root_dir, "erdos_renyi", graph_name+".pt"))
        elif graph_name.startswith("sbm"):
            torch.save(predict_layout, os.path.join(output_root_dir, "sbm", graph_name+".pt"))

        log_string = " graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item()) + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "predict_log.txt"), log_string + "\n")
    
    
    log_string = "average loss: "+str(average_predict_loss)+" avg pre energy: "+str(avg_predict_pre_energy)+" avg truth energy: "+str(avg_predict_tru_energy)    
    write_log(os.path.join(log_dir, "predict_log.txt"), log_string+"\n\n")    
    print(log_string)

def main():
    if not no_train:
        best_val_loss = np.inf
        best_epoch = -1
        for epoch in tqdm(range(epochs), desc="epoch", ncols=80):
            average_val_loss = train(epoch, best_val_loss)
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_epoch = epoch
        log_string = "Best model epoch: "+str(best_epoch)+" average val loss: "+str(best_val_loss)
        print(log_string)
        write_log(os.path.join(log_dir, "val_log.txt"), log_string+"\n\n")        
    
    if not no_test:
        average_test_loss = test()
        print("Test average loss: ", average_test_loss)
    
    if not no_predict:
        predict(model_load_path)
    
if __name__ == "__main__":
    main()