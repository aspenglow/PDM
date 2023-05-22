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
    # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #     dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# Create the 3D figure
fig = plt.figure()

graph_name="sbm89"
graph_path = './graphs_sbm_2/sbm/' + graph_name + '.pkl'
layout_path = 'layouts_sbm_2/sbm/' + graph_name + '.pt'
output_path = 'outputs_sbm_2/sbm/' + graph_name + '.pt'

G = nx.read_gpickle(graph_path)
output = torch.load(output_path, map_location=torch.device('cpu'))[0]
N = output.shape[0]

pos = output.detach().numpy()
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
# Create the 3D figure
ax2 = fig.add_subplot(122, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax2.scatter(*node_xyz[:int(N/2)].T, s=50, c="b")
ax2.scatter(*node_xyz[int(N/2):].T, s=50, c="r")
ax2.set_title("Predict layout")

# Plot the edges
for vizedge in edge_xyz:
    ax2.plot(*vizedge.T, color="tab:gray")

_format_axes(ax2)

print(N)

G = nx.read_gpickle(graph_path)
layout = torch.load(layout_path, map_location=torch.device('cpu'))
N = layout.shape[0]
R = orthogonal_procrustes(layout.to('cpu').detach().numpy(), output.to('cpu').detach().numpy())[0]
R = torch.from_numpy(R)
layout = torch.matmul(layout, R)

pos = layout.detach().numpy()
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
# nx.draw_networkx_nodes(G, layout.detach().numpy())

ax1 = fig.add_subplot(121, projection="3d")
ax1.set_title("FDL layout")

# Plot the nodes - alpha is scaled by "depth" automatically
ax1.scatter(*node_xyz[:int(N/2)].T, s=50, c="b")
ax1.scatter(*node_xyz[int(N/2):].T, s=50, c="r")

# Plot the edges
for vizedge in edge_xyz:
    ax1.plot(*vizedge.T, color="tab:gray")

_format_axes(ax1)
fig.tight_layout()
Adj = nx.to_numpy_matrix(G)
Adj = torch.from_numpy(Adj)
adjacency_list = torch.where(torch.triu(Adj))
predict_energy = energy(output, adjacency_list, N)
truth_energy = energy(layout, adjacency_list, N)

print("predict energy: ", predict_energy.item(), "truth energy: ", truth_energy.item())
plt.show()