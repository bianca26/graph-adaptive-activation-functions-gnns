# Bianca Iancu, bianca.iancu026@gmail.com
"""
Module for implementing the localized graph-adaptive activation function.

Activation Functions - Nonlinearities (nn.Module)

MaxDistributedActivation: Creates a graph-adaptive max activation function layer
MedianDistributedActivation: Creates a graph-adaptive median activation function layer

"""

import math
import numpy as np
import torch
import torch.nn as nn

import Utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number


class MaxGAActivation(nn.Module):

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
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        maxNeighborhoodSizes = []

        # compute the one-hop neighborhood - since the nonlinearity only operates in the one-hop neighborhood
        thisNeighborhood = graphTools.computeNeighborhood(
                np.array(self.S.cpu()), k, outputType='matrix')
        # compute the k-hop neighborhood
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
        xK = activation(x)  # we first account for the signal value of the node itself
        xK = xK.unsqueeze(3)  # extra dimension added for concatenation

        for _ in range(1, self.K + 1):
            x = x.reshape([batchSize, 1, dimNodeSignals, self.N])  # B x 1 x G x N
            x = torch.matmul(x, self.S)  # B x 1 x G x N
            x = torch.squeeze(x, 1)  # B x G x N

            x_aux = x.unsqueeze(3)  # B x F x N x 1

            x_aux = x_aux.repeat([1, 1, 1, self.maxNeighborhoodSizes[0]])
            gatherNeighbor = self.neighborhood[0].reshape(
                [1,
                 1,
                 self.N,
                 self.maxNeighborhoodSizes[0]]
            )
            gatherNeighbor = gatherNeighbor.repeat([batchSize,
                                                    dimNodeSignals,
                                                    1,
                                                    1])
            # Get all the neighbors in line
            xNeighbors = torch.gather(x_aux, 2, gatherNeighbor.long().cuda())
            #COmpute the maximum along this dimension
            v, _ = torch.max(xNeighbors, dim=3)
            v = v.unsqueeze(3)  # to concatenate with xK
            xK = torch.cat((xK, v), 3)
        # Multiply each k-hop max by the corresponding trainable parameter
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



class MedianGAActivation(nn.Module):

    def __init__(self, K = 1):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        self.weight = nn.parameter.Parameter(torch.Tensor(1, self.K + 1))
        #self.masks = 'None' # no mask yet
        # Create parameters:
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
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []

        # compute the one-hop neighborhood - since the nonlinearity only operates in the one-hop neighborhood
        thisNeighborhood = graphTools.computeNeighborhood(
            np.array(self.S.cpu()), k, outputType='matrix')
        # compute the k-hop neighborhood
        neighborhood.append(torch.tensor(thisNeighborhood))
        self.neighborhood = neighborhood


    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N

        self.S = self.S.reshape([1, 1, self.N, self.N])  # 1 x 1 x N x N

        activation = nn.ReLU()
        xK = activation(x)  # we first account for the signal value of the node itself
        xK = xK.unsqueeze(3)  # extra dimension added for concatenation

        for _ in range(1, self.K+1):
            x = x.reshape([batchSize, 1, dimNodeSignals, self.N])  # B x 1 x G x N
            x = torch.matmul(x, self.S)  # B x 1 x G x N
            x = torch.squeeze(x, 1) # B x G x N

            #Retrieve one-hop neighborhoods of all nodes
            kHopNeighborhood = self.neighborhood[0]

            #Initialize the vector that will contain the k-hop median for every node
            kHopMedian = torch.empty(0)

            #Iterate over the nodes; neighborhoods are lists of lists
            for n in range(self.N):
                nodeNeighborhood = torch.tensor(np.array(kHopNeighborhood[n]))
                neighborhoodLen = len(nodeNeighborhood)
                # Reshaping the node neighborhood for the gather operation
                gatherNode = nodeNeighborhood.reshape([1, 1, neighborhoodLen])
                gatherNode = gatherNode.repeat([batchSize, dimNodeSignals, 1])
                # Gathering signal values in the node neighborhood
                xNodeNeighbors = torch.gather(x, 2, gatherNode.long().cuda())
                # Computing the median in the neighborhood
                nodeMedian,_ = torch.median(xNodeNeighbors, dim = 2,
                                            keepdim=True)
                # Concatenating k-hop medians node by node
                kHopMedian = torch.cat([kHopMedian.cuda(),nodeMedian.cuda()],2)
            kHopMedian = kHopMedian.unsqueeze(3)
            xK = torch.cat([xK,kHopMedian],3)
        # Multiplying each k-hop median by corresponding trainable parameter
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        out = out.reshape([batchSize,dimNodeSignals,self.N])
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