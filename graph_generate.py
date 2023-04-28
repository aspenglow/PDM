import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Generate random graphs.')
    parse.add_argument('--num_er', type=int, default=500, help='number of ER graph.')
    parse.add_argument('--num_sbm', type=int, default=500, help='number of SBM graph.')
    args = parse.parse_args()
    return args


def generate_erdos_renyi_graph(save_dir='graphs/erdos_renyi', num_graphs = 500, n_range=[100,1000], p=0.1):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(num)):
        n = np.random.randint(n_range[0], n_range[1])
        g = nx.random_graphs.erdos_renyi_graph(n,p)
        while not nx.is_connected(g):
            g = nx.random_graphs.erdos_renyi_graph(n,p)
        nx.write_gpickle(g, os.path.join(save_dir, "er"+str(i+1)+".pkl"))
    if num_graphs == 1:
        return g
        
def generate_sbm_graph(save_dir='graphs/sbm', num_graphs=500, num_blocks=2, node_range=[50,500], p_in=0.2, p_betw=0.02):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(num_graphs)):
        probs = p_betw * np.ones((num_blocks, num_blocks))
        sizes = []
        for j in range(num_blocks):
            sizes += [np.random.randint(node_range[0], node_range[1])]
            probs[j,j] += (p_in - p_betw) 
        
        g = nx.stochastic_block_model(sizes, probs)
        while not nx.is_connected(g):
            g = nx.stochastic_block_model(sizes, probs)
        nx.write_gpickle(g, os.path.join(save_dir, "sbm"+str(i+1)+".pkl"))
    if num_graphs == 1:
        return g
        

if __name__ == "__main__":
    args = parse_args()
    generate_erdos_renyi_graph(num_graphs=args.num_er)
    generate_sbm_graph(num_graphs=args.num_sbm)
    
