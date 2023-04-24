import numpy as np
import time
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


# measure time
class tictoc():
    def __init__(self):
        self.prev = 0
        self.now = 0
    def tic(self):
        self.prev = time.time()
    def toc(self):
        self.now = time.time()
        #print( "dt(s) = %.3g" %(self.now - self.prev))
        t = self.now - self.prev
        #self.prev = time.time()
        return t
 
## sparse matrix formula   
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#model
def c_kdtree(x, r):
    tree = cKDTree(x.detach().numpy())
    return tree.query_pairs(r,output_type = 'ndarray')
        
def Distances_kdtree(X, pairs):
        X1 = X[pairs[:,0]]
        X2 = X[pairs[:,1]]
        dX = torch.sum((X1-X2)**2, axis = -1) 
        
        return dX
    
#optimizer 
def difference(r):
    return (max(r)-min(r))/max(r)

def early_stopping(metric_list,
            small_window = 32,
            big_window = 1000,
            stop_delta_ratio = 1e-3, verbose=False):
    if len(metric_list) < 2*small_window:
        return False
    # check if chenges within big window and small window are smaller then the ratio
    big_window = max(big_window, 2*small_window)
    last = np.mean(metric_list[-small_window:])
    dl_small =  abs(last - np.mean(metric_list[-2*small_window:-small_window]))
    idx = max(0,len(metric_list)-big_window)
    dl_big = abs(last - np.mean(metric_list[idx:idx+small_window]))
    ratio = dl_small / dl_big
    if verbose: 
        print(f'step: {len(metric_list)}, Loss change ratio: {ratio:.3g}', end='\r')
        # print(f'Loss change ratio: {ratio:.3g}', end='\r')
    return ratio < stop_delta_ratio 
    