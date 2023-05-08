#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   latentgnn.py
@Time    :   2019/05/27 12:05:11
@Author  :   Songyang Zhang 
@Version :   1.0
@Contact :   sy.zhangbuaa@hotmail.com
@License :   (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University
@Desc    :   None
'''

import torch

from latentgnn_v1 import LatentGNN

def test_latentgnn():
    network = LatentGNN(in_features=20,
                        out_features=3,
                        latent_dims=[100,100],
                        channel_stride=10,
                        num_kernels=2,
                        mode='symmetric',
                        graph_conv_flag=False)
    
    dump_inputs = torch.rand((8, 500, 20))
    print(str(network))
    output = network(dump_inputs)
    print(output.shape)


if __name__ == "__main__":
    test_latentgnn()
    # test_group_latentgnn()