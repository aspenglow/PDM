import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

def generate_erdos_renyi_graph(save_dir='graphs/erdos_renyi', num = 500, n_range=[100,1000], p=0.1):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i in tqdm(range(num)):
        n = np.random.randint(n_range[0], n_range[1])
        g = nx.random_graphs.erdos_renyi_graph(n,p)
        if not nx.is_connected(g):
            i -= 1
            continue
        nx.write_gpickle(g, os.path.join(save_dir, "er"+str(i+1)+".pkl"))
        
def generate_sbm_graph(save_dir='graphs/sbm', num_graphs=500, num_blocks=2, node_range=[50,500], p_in=0.2, p_betw=0.05):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i in tqdm(range(num_graphs)):
        probs = p_betw * np.ones((num_blocks, num_blocks))
        sizes = []
        for j in range(num_blocks):
            sizes += [np.random.randint(node_range[0], node_range[1])]
            probs[j,j] += (p_in - p_betw) 
        
        g = nx.stochastic_block_model(sizes, probs)
        if not nx.is_connected(g):
            i -= 1
            continue
        nx.write_gpickle(g, os.path.join(save_dir, "sbm"+str(i+1)+".pkl"))
        

if __name__ == "__main__":
    generate_erdos_renyi_graph(num=500)
    generate_sbm_graph(num_graphs=500)
    
