import torch

def graph_to_edge_list(G):
  # TODO: Implement the function that returns the edge list of
  # an nx.Graph. The returned edge_list should be a list of tuples
  # where each tuple is a tuple representing an edge connected
  # by two nodes.
 
  edge_list = []
 
  ############# Your code here ############
  for e in G.edges():
      edge_list.append(e)
  #########################################
 
  return edge_list
 
def edge_list_to_tensor(edge_list):
  # TODO: Implement the function that transforms the edge_list to
  # tensor. The input edge_list is a list of tuples and the resulting
  # tensor should have the shape [2 x len(edge_list)].
 
  edge_index = torch.tensor([])
 
  ############# Your code here ############
  edge_index = torch.LongTensor(edge_list).t()
  #########################################
 
  return edge_index
