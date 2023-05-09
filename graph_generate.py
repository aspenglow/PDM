import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Generate random graphs.')
    parse.add_argument('--save-dir-er', type=str, default='graphs/erdos_renyi', help='Save dir of ER graph.')
    parse.add_argument('--save-dir-sbm', type=str, default='graphs/sbm', help='Save dir of sbm graph.')
    parse.add_argument('--num-er', type=int, default=500, help='Number of ER graph.')
    parse.add_argument('--num-sbm', type=int, default=500, help='Number of SBM graph.')
    parse.add_argument('--index-start-at-er', type=int, default=1, help='Start index of ER graph.')
    parse.add_argument('--index-start-at-sbm', type=int, default=1, help='Start index of sbm graph.')
    args = parse.parse_args()
    return args


def generate_erdos_renyi_graph(save_dir='graphs/erdos_renyi', num_graphs=500, index_start_at=1, n_range=[100,1000], p=0.1):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(index_start_at, index_start_at+num_graphs)):
        n = np.random.randint(n_range[0], n_range[1])
        g = nx.random_graphs.erdos_renyi_graph(n,p)
        while not nx.is_connected(g):
            g = nx.random_graphs.erdos_renyi_graph(n,p)
        nx.write_gpickle(g, os.path.join(save_dir, "er"+str(i)+".pkl"))
    if num_graphs == 1:
        return g
        
def generate_sbm_graph(save_dir='graphs/sbm', num_graphs=500, index_start_at=1, num_blocks=2, node_range=[50,500], p_in=0.2, p_betw=0.02):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(index_start_at, index_start_at+num_graphs)):
        probs = p_betw * np.ones((num_blocks, num_blocks))
        sizes = []
        for j in range(num_blocks):
            sizes += [np.random.randint(node_range[0], node_range[1])]
            probs[j,j] += (p_in - p_betw) 
        
        g = nx.stochastic_block_model(sizes, probs)
        while not nx.is_connected(g):
            g = nx.stochastic_block_model(sizes, probs)
        nx.write_gpickle(g, os.path.join(save_dir, "sbm"+str(i)+".pkl"))
    if num_graphs == 1:
        return g
        

if __name__ == "__main__":
    args = parse_args()
    generate_erdos_renyi_graph(save_dir=args.save_dir_er, num_graphs=args.num_er, index_start_at=args.index_start_at_er)
    generate_sbm_graph(save_dir=args.save_dir_sbm, num_graphs=args.num_sbm, index_start_at=args.index_start_at_sbm)
    
