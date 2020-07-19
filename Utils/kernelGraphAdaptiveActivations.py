# Bianca Iancu, bianca.iancu026@gmail.com
"""
Module for implementing the kernel graph-adaptive activation function.

Activation Functions - Nonlinearities (nn.Module)

KernelActivation: Creates a graph-adaptive kernel activation function layer. The kernel graph-adaptive
activation function employs the Gaussian kernel.

"""


import math
import numpy as np
import torch
import torch.nn as nn

import Utils.graphTools as graphTools

#Define first the kernel function that we are going to use

def gaussian_kernel(tensor_node, tensor_neighors):
    """

    :param tensor_node: tensor of the value of the node, repeated for as many times as the number
    of one-hope neighbors of the node
    :param tensor_neighors: tensor containing the signal values at the neighbors of the node
    :return: Gaussian kernel value
    """
    nBatches = list(tensor_neighors.size())[0]
    nFeatures = list(tensor_neighors.size())[1]

    two_sigma_squared = 2 * (0.1 * 0.1)

    res_sum = torch.sum(torch.pow(tensor_node - tensor_neighors, 2))
    res_sqrt = (res_sum / two_sigma_squared) * (-1)
    res = torch.exp(res_sqrt)

    final_kernel_value = res.repeat([nBatches, nFeatures, 1])  # This should be of size [B, F, 1]

    return final_kernel_value



class KernelGAActivation(nn.Module):

    def __init__(self, K=1):

        super().__init__()
        assert K > 0  # range has to be greater than 0
        self.K = K
        self.S = None  # no GSO assigned yet
        self.N = None  # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None'  # no neighborhoods calculated yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1, self.K + 1))
        # Initialize parameters
        self.reset_parameters()

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        # The neighborhood matrix has to be a tensor of shape
        # nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        maxNeighborhoodSizes = []

        # compute the one-hop neighborhood - since the nonlinearity only operates in the one-hop neighborhood
        thisNeighborhood = graphTools.computeNeighborhood_without_current(np.array(self.S.cpu()), 1, outputType='matrix')
        neighborhood.append(torch.tensor(thisNeighborhood))
        maxNeighborhoodSizes.append(thisNeighborhood.shape[1])
        self.maxNeighborhoodSizes = maxNeighborhoodSizes
        self.neighborhood = neighborhood


    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N

        self.S = self.S.reshape([1, 1, self.N, self.N])  # 1 x 1 x N x N

        activation = nn.ReLU()
        xK = activation(x) # we first account for the signal value of the node itself
        xK = xK.unsqueeze(3)  # extra dimension added for concatenation ahead

        for _ in range(1, self.K + 1):
            kHopNeighborhood = self.neighborhood[0]
            kHopKernel = torch.empty(0)

            #Iterate over all the nodes in the graph
            for n in range(self.N):
                # Tensor for neighborhood nodes
                nodeNeighborhood = torch.tensor(np.array(kHopNeighborhood[n]))
                neighborhoodLen = len(nodeNeighborhood)
                gatherNode = nodeNeighborhood.reshape([1, 1, neighborhoodLen])
                gatherNode = gatherNode.repeat([batchSize, dimNodeSignals, 1])
                xNodeNeighbors = torch.gather(x, 2,
                                              gatherNode.long().cuda())  # This gathers the signal of the neighbrs of the current node
                # in dimension BxFxLenNeigh


                # Tensor for current node
                currentNode = torch.tensor(np.asarray([n], dtype=np.float32))
                currentNodeLen = len(currentNode)
                gatherNodeCurrent = currentNode.reshape([1, 1, currentNodeLen])
                gatherNodeCurrent = gatherNodeCurrent.repeat([batchSize, dimNodeSignals, 1])
                xCurrentNode = torch.gather(x, 2, gatherNodeCurrent.long().cuda())  # This gathers the signal of the current node
                # in dimension BxFx1
                xCurrentNode = xCurrentNode.repeat([1, 1, neighborhoodLen]) #dimension BxFxLenNeigh


                nodeKernel = gaussian_kernel(xCurrentNode, xNodeNeighbors)
                kHopKernel = torch.cat([kHopKernel.cuda(), nodeKernel.cuda()], 2)

            kHopKernel = kHopKernel.unsqueeze(3)
            xK = torch.cat([xK, kHopKernel], 3)

        # multiply each k-hop max by corresponding trainable parameter
        out = torch.matmul(xK, self.weight.unsqueeze(2))
        out = out.reshape([batchSize, dimNodeSignals, self.N])
        return out

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.weight.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        if self.neighborhood is not None:
            reprString = "neighborhood stored"
        else:
            reprString = "NO neighborhood stored"
        return reprString

