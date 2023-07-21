import os
import torch
import networkx as nx

import sys
sys.path.append(os.getcwd())
from utils.utils_neulay import *
from neulay.neulay_model import *
from graph_generate import *

import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

from utils.utils_neulay import energy 

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis):
        dim.set_ticks([])
    # Set axes labels
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
 
colors = ['b','r','g','c','m','y','k','brown']   
def draw_layout(G, layout, ax, title="layout"):
    
    N = layout.shape[0]

    pos = layout.detach().numpy()
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
    block_info = np.array([v[1]["block"] for v in G.nodes(data=True)])
    for i in sorted(G):
        color = colors[block_info[i]]
        ax.scatter(*node_xyz[i].T, s=30, c=color)
    
    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", linewidth=0.2, zorder=0)
        
    _format_axes(ax)
    ax.set_title(title, fontsize=24)