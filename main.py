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
from latentgnn.latentgnn_v1 import LatentGNN
from latentgnn.baseline import Baseline_GCN, Baseline_FC
from utils.dataset import LayoutDataset
from utils.utils import load_data, two_norm_distance, energy_loss, write_log
from utils.utils_neulay import energy 


import argparse

parser = argparse.ArgumentParser(
    'Train LatentGNN model to predict final layout from different graphs.')
parser.add_argument('--experiment-root-dir', type=str, default="experiment_template",
                    help='Root path for experiment.')
parser.add_argument('--graph-root-dir', type=str, default="graphs_sbm_3/",
                    help='Root path to load graphs.')
parser.add_argument('--layout-root-dir', type=str, default="layouts_sbm_3/",
                    help='Root path to load final layouts.')
parser.add_argument('--predict-graph-root-dir', type=str, default=None,
                    help='Root path to load graph to be predict.')
parser.add_argument('--predict-layout-root-dir', type=str, default=None,
                    help='Root path to load layouts corresponding to the graph to be predict.')
parser.add_argument('--output-root-dir', type=str, default="outputs_sbm_3/",
                    help='Root path to save predicted final layouts.')
parser.add_argument('--model', type=str, default="latentgnn",
                    help='Which kind of model to use (latentgnn/baseline_GCN/baseline_FC).')
parser.add_argument('--model-save-dir', type=str, default="model_sbm_3/",
                    help='Dir to save best model.')
parser.add_argument('--model-load-path', type=str, default=None,
                    help='Path to load trainer models. "None" means train a new model. ')
parser.add_argument('--log-dir', type=str, default="log_sbm_3",
                    help='Dir to save train validation and test logs.')
parser.add_argument('--train-set-ratio', type=float, default=0.7,
                    help='Ratio of train set.')
parser.add_argument('--val-set-ratio', type=float, default=0.2,
                    help='Ratio of validation set.')
parser.add_argument('--position-encoding-dim', type=int, default=60,
                    help='Dim of graph position encoding as input encoding.')
parser.add_argument('--latent-dims', nargs='+', type=int, default=[4,4,4],
                    help='Number of nodes for each latent graph layer.')
parser.add_argument('--layout-dim', type=int, default=2)
parser.add_argument('--channel-multi', type=int, default=20,
                    help='How many multiple channels in latent space compare with visible space.')
parser.add_argument('--mode', type=str, default="asymmetric",
                    help='Which kind of latentGNN to use (symmetric/asymmetric).')
parser.add_argument('--loss', type=str, default="coordinate_diff",
                    help='Which kind of loss function to use (energy/coordinate_diff).')
parser.add_argument('--epochs', type=int, default=60,
                    help='Numer of epoch to train.')
parser.add_argument('--init-lr', type=float, default=0.0005,
                    help='Parameter to decay the learning rate per epoch.')
parser.add_argument('--scheduler-step-size', type=int, default=1,
                    help='Parameter to decay the learning rate per epoch.')
parser.add_argument('--gamma', type=float, default=0.98,
                    help='Parameter to decay the learning rate per epoch.')
parser.add_argument('--re-split-dataset', action="store_true",
                    help='Use gpu for training.')
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

experiment_root_dir = args.experiment_root_dir
graph_root_dir = os.path.join(experiment_root_dir, args.graph_root_dir)
layout_root_dir = os.path.join(experiment_root_dir, args.layout_root_dir)
predict_graph_root_dir = os.path.join(experiment_root_dir, args.predict_graph_root_dir)
predict_layout_root_dir = os.path.join(experiment_root_dir, args.predict_layout_root_dir)
output_root_dir = os.path.join(experiment_root_dir, args.output_root_dir)
model_save_dir = os.path.join(experiment_root_dir, args.model_save_dir)
if args.model_load_path is not None:
    model_load_path = os.path.join(experiment_root_dir, args.model_load_path)
else:
    model_load_path = None

train_set_ratio = args.train_set_ratio
val_set_ratio = args.val_set_ratio
assert(train_set_ratio + val_set_ratio < 1.0)
test_set_ratio = 1.0 - train_set_ratio - val_set_ratio
dataset_ratio = [train_set_ratio, val_set_ratio, test_set_ratio]

encoding_dim = args.position_encoding_dim
latent_dims = args.latent_dims
layout_dim = args.layout_dim
channel_multi = args.channel_multi

epochs = args.epochs
init_lr = args.init_lr
scheduler_step_size = args.scheduler_step_size
gamma = args.gamma
log_dir = os.path.join(experiment_root_dir, args.log_dir)
os.makedirs(log_dir, exist_ok=True)

no_train = args.no_train
no_test = args.no_test
no_predict = args.no_predict

# Load dataloader
data_loader, train_loader, validation_loader, test_loader = load_data(experiment_root_dir, graph_root_dir, layout_root_dir, dataset_ratio, re_split_dataset=args.re_split_dataset)

# Initialize models
if args.model == "latentgnn":
    model = LatentGNN(in_features=encoding_dim, out_features=layout_dim, latent_dims=latent_dims, channel_multi=channel_multi, mode=args.mode)  
    print("model: latentgnn")
elif args.model == "baseline_GCN":
    model = Baseline_GCN(in_features=encoding_dim, out_features=layout_dim)  
    print("model: baseline GCN")
elif args.model == "baseline_FC":
    model = Baseline_FC(in_features=encoding_dim, out_features=layout_dim)

if model_load_path is not None:
    model.load_state_dict(torch.load(model_load_path))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

lapEncoding = LapEncoding(dim=encoding_dim)

def train(epoch, best_energy_relative_diff) -> float:
    # Training
    model.train()
    average_train_loss = 0.
    avg_train_pre_energy = 0.
    avg_train_tru_energy = 0.
    avg_energy_relative_diff = 0.
    
    for batch_idx, (Adj, layout, graph_name) in enumerate(train_loader):
        optimizer.zero_grad()
        N = Adj.size(1)
        L = torch.eye(N) - Adj[0]
        Adj = Adj.to(device)
        L = L.to(device)
        layout = layout.to(device)
        encoding = lapEncoding.compute_pe(L).unsqueeze(0)
        
        predict_layout = model(Adj, encoding)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        energy_relative_diff = energy_loss(predict_energy, truth_energy)
        
        
        # print("predict: ", predict_energy, " truth: ", truth_energy)
        
        if args.loss == "energy":
            loss = energy_relative_diff
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = two_norm_distance(predict_layout, layout)

        loss.backward()
        
        average_train_loss += loss.item() / len(train_loader)
        avg_train_pre_energy += predict_energy.item() / len(train_loader)
        avg_train_tru_energy += truth_energy.item() / len(train_loader)
        avg_energy_relative_diff += energy_relative_diff.item() / len(train_loader)
        
        graph_name = graph_name[0]
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item())[:7]+" p_energy: "+str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "train_log.txt"), log_string + "\n")
    
    log_string = "epoch "+str(epoch)+" average ERD: "+str(avg_energy_relative_diff)[:7]+" avg pre energy: "+str(avg_train_pre_energy)+" avg truth energy: "+str(avg_train_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
    write_log(os.path.join(log_dir, "train_log.txt"), log_string+"\n\n")
    write_log(os.path.join(log_dir, "train_log_summary.txt"), log_string+"\n")
    
    optimizer.step()
    scheduler.step()
    
    # Validation
    model.eval()
    avg_val_loss = 0.
    avg_val_pre_energy = 0.
    avg_val_tru_energy = 0.
    avg_energy_relative_diff = 0.
    for batch_idx, (Adj, layout, graph_name) in enumerate(validation_loader):
        N = Adj.size(1)
        L = torch.eye(N) - Adj[0]
        Adj = Adj.to(device)
        L = L.to(device)
        layout = layout.to(device)
        encoding = lapEncoding.compute_pe(L).unsqueeze(0)
        
        predict_layout = model(Adj, encoding)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        energy_relative_diff = energy_loss(predict_energy, truth_energy)
        if args.loss == "energy":
            loss = energy_relative_diff
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = two_norm_distance(predict_layout, layout)
        
        avg_val_loss += loss.item() / len(validation_loader)
        avg_val_pre_energy += predict_energy.item() / len(validation_loader)
        avg_val_tru_energy += truth_energy.item() / len(validation_loader)
        avg_energy_relative_diff += energy_relative_diff.item() / len(validation_loader)
        
        graph_name = graph_name[0]
        log_string = "epoch "+str(epoch)+" graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" loss: "+ str(loss.item())[:7] + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "val_log.txt"), log_string + "\n")
    
    log_string = "epoch "+str(epoch)+" average ERD: "+str(avg_energy_relative_diff)[:7]+" avg pre energy: "+str(avg_val_pre_energy)+" avg truth energy: "+str(avg_val_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
    write_log(os.path.join(log_dir, "val_log.txt"), log_string+"\n\n")
    write_log(os.path.join(log_dir, "val_log_summary.txt"), log_string+"\n")
    
    if avg_energy_relative_diff < best_energy_relative_diff:
        write_log(os.path.join(log_dir, "val_log.txt"), "Better model! saving..."+"\n\n")
        write_log(os.path.join(log_dir, "val_log_summary.txt"), "Better model! saving..."+"\n\n")
        if model_save_dir is not None:
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, "model.pt"))
    return avg_energy_relative_diff
   
    
def test(model_load_path: str=None) -> float:
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
    model.eval()
    
    average_test_loss = 0.
    avg_test_pre_energy = 0.
    avg_test_tru_energy = 0.
    avg_energy_relative_diff = 0.
    
    for batch_idx, (Adj, layout, graph_name) in enumerate(test_loader):
        N = Adj.size(1)
        L = torch.eye(N) - Adj[0]
        Adj = Adj.to(device)
        L = L.to(device)
        layout = layout.to(device)
        encoding = lapEncoding.compute_pe(L).unsqueeze(0)
        
        predict_layout = model(Adj, encoding)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        energy_relative_diff = energy_loss(predict_energy, truth_energy)
        if args.loss == "energy":
            loss = energy_relative_diff
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = two_norm_distance(predict_layout, layout)
        
        average_test_loss += loss.item() / len(test_loader)
        avg_test_pre_energy += predict_energy.item() / len(test_loader)
        avg_test_tru_energy += truth_energy.item() / len(test_loader)
        avg_energy_relative_diff += energy_relative_diff.item() / len(test_loader)
        
        graph_name = graph_name[0]
        log_string = " graph "+str(batch_idx)+" "+graph_name+" N_nodes: "+str(N)+" ERD: "+str(energy_relative_diff.item())[:7] \
            +" loss: "+ str(loss.item())[:7]+" p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "test_log.txt"), log_string + "\n")
    
    log_string = "Average ERD: "+str(avg_energy_relative_diff)+" avg pre energy: "+str(avg_test_pre_energy)+" avg truth energy: "+str(avg_test_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
    write_log(os.path.join(log_dir, "test_log.txt"), log_string+"\n\n")
    return avg_energy_relative_diff

def predict(model_load_path: str=None):
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_dir, "model.pt")))
    model.eval()
    if predict_graph_root_dir is not None and predict_layout_root_dir is not None:
        dataset = LayoutDataset(graph_root_dir=predict_graph_root_dir, layout_root_dir=predict_layout_root_dir)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print("new predict graphs.")
    
    average_predict_loss = 0.
    avg_predict_pre_energy = 0.
    avg_predict_tru_energy = 0.
    avg_energy_relative_diff = 0.
    
    block_avg_predict_pre_energy = 0.
    block_avg_predict_tru_energy = 0.
    block_avg_energy_relative_diff = 0.
    
    dirs = os.listdir(layout_root_dir)
    for dir in dirs:
        os.makedirs(os.path.join(output_root_dir, dir), exist_ok=True)

    for batch_idx, (Adj, layout, graph_name) in enumerate(data_loader):
        
        N = Adj.size(1)
        L = torch.eye(N) - Adj[0]
        Adj = Adj.to(device)
        L = L.to(device)
        layout = layout.to(device)
        encoding = lapEncoding.compute_pe(L).unsqueeze(0)
        
        predict_layout = model(Adj, encoding)
        R = orthogonal_procrustes(predict_layout[0].to('cpu').detach().numpy(), layout[0].to('cpu').detach().numpy())[0]
        R = torch.from_numpy(R).unsqueeze(0).to(device)
        predict_layout = torch.bmm(predict_layout, R)
        adjacency_list = torch.where(torch.triu(Adj[0]))
        predict_energy = energy(predict_layout[0], adjacency_list, N)
        truth_energy = energy(layout[0], adjacency_list, N)
        energy_relative_diff = energy_loss(predict_energy, truth_energy)
        if args.loss == "energy":
            loss = energy_relative_diff
        elif args.loss == "coordinate_diff":
            R = orthogonal_procrustes(layout[0].to('cpu').detach().numpy(), predict_layout[0].to('cpu').detach().numpy())[0]
            R = torch.from_numpy(R).unsqueeze(0).to(device)
            layout = torch.bmm(layout, R)
            loss = two_norm_distance(predict_layout, layout)
        
        average_predict_loss += loss.item() / len(data_loader)
        avg_predict_pre_energy += predict_energy.item() / len(data_loader)
        avg_predict_tru_energy += truth_energy.item() / len(data_loader)
        avg_energy_relative_diff += energy_relative_diff.item() / len(data_loader)
        
        block_avg_predict_pre_energy += predict_energy.item() / 50
        block_avg_predict_tru_energy += truth_energy.item() / 50
        block_avg_energy_relative_diff += energy_relative_diff.item() / 50
        
        graph_name: str = graph_name[0]
        graph_dir, graph_name = os.path.split(graph_name)
        
        # if graph_name.startswith("er"):
        #     torch.save(predict_layout, os.path.join(output_root_dir, graph_dir, graph_name+".pt"))
        # elif graph_name.startswith("sbm"):
        # os.makedirs(os.path.join(output_root_dir, graph_dir), exist_ok=True)
        # torch.save(predict_layout, os.path.join(output_root_dir, graph_dir, graph_name+".pt"))

        log_string = " graph "+str(batch_idx)+" "+graph_dir+"/"+graph_name+" N_nodes: "+str(N)+" ERD: " + str(energy_relative_diff.item())[:7]+" loss: "+ str(loss.item())[:7] + " p_energy: " + str(predict_energy.item()) + " t_energy: " + str(truth_energy.item())
        write_log(os.path.join(log_dir, "predict_log.txt"), log_string + "\n")
        
        if batch_idx % 50 == 49:
            log_string = "Average block ERD: "+str(block_avg_energy_relative_diff)[:7]+" avg block pre energy: "+str(block_avg_predict_pre_energy)+" block avg truth energy: "+str(block_avg_predict_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
            write_log(os.path.join(log_dir, "predict_log.txt"), log_string+"\n\n")    
    
            block_avg_predict_pre_energy = 0.
            block_avg_predict_tru_energy = 0.
            block_avg_energy_relative_diff = 0.
    
    
    log_string = "Average ERD: "+str(avg_energy_relative_diff)[:7]+" avg pre energy: "+str(avg_predict_pre_energy)+" avg truth energy: "+str(avg_predict_tru_energy)+" lr: "+str(scheduler.get_last_lr()[0])
    write_log(os.path.join(log_dir, "predict_log.txt"), log_string+"\n\n")    
    print(log_string)

def main():
    print("latent_dims: ", latent_dims, "encoding_dim: ", encoding_dim, "channels_multi: ", channel_multi)
    print("loss function: ", args.loss)
    if not no_train:
        best_energy_relative_diff = np.inf
        best_epoch = -1
        for epoch in tqdm(range(epochs), desc="epoch", ncols=80):
            avg_energy_relative_diff = train(epoch, best_energy_relative_diff)
            if avg_energy_relative_diff < best_energy_relative_diff:
                best_energy_relative_diff = avg_energy_relative_diff
                best_epoch = epoch
        log_string = "Best model epoch: "+str(best_epoch)+" average energy relative diff: "+str(best_energy_relative_diff)
        print(log_string)
        write_log(os.path.join(log_dir, "val_log_summary.txt"), log_string+"\n\n")        
    
    if not no_test:
        test_avg_energy_relative_diff = test()
        print("Test average energy relative loss: ", test_avg_energy_relative_diff)
    
    if not no_predict:
        predict(model_load_path)
    
if __name__ == "__main__":
    main()