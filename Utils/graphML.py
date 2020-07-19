# 2018/11/01~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu.
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

LSIGF: Applies a linear shift-invariant graph filter


Activation Functions - Nonlinearities (nn.Module)

MaxLocalActivation: Creates a localized max activation function layer
MedianLocalActivation: Creates a localized median activation function layer
NoActivation: Creates a layer for no activation function

Summarizing Functions - Pooling (nn.Module)

NoPool: No summarizing function.
"""

import math
import numpy as np
import torch
import torch.nn as nn

import Utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

# WARNING: Only scalar bias.

def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    #print(f"Graph Filter {np.shape(x)}")
    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute again to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y


class MaxLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/15
    """
    MaxLocalActivation creates a localized activation function layer on graphs

    Initialization:

        MaxLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized max activation function layer

    Add graph shift operator:

        MaxLocalActivation.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MaxLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K = 1):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
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
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(self.S.cpu()), k, outputType='matrix')
            # compute the k-hop neighborhood
            neighborhood.append(torch.tensor(thisNeighborhood))
            maxNeighborhoodSizes.append(thisNeighborhood.shape[1])
        self.maxNeighborhoodSizes = maxNeighborhoodSizes
        self.neighborhood = neighborhood

    def forward(self, x):
        #print(f"forward in activation function {x.size()}")
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's it is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        x = x.unsqueeze(3) # B x F x N x 1
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        for k in range(1,self.K+1):
            x_aux = x.repeat([1, 1, 1, self.maxNeighborhoodSizes[k-1]])
            gatherNeighbor = self.neighborhood[k-1].reshape(
                                                [1,
                                                 1,
                                                 self.N,
                                                 self.maxNeighborhoodSizes[k-1]]
                                                )
            gatherNeighbor = gatherNeighbor.repeat([batchSize, 
                                                    dimNodeSignals,
                                                    1,
                                                    1])
            # And finally we're in position of getting all the neighbors in line
            xNeighbors = torch.gather(x_aux, 2, gatherNeighbor.long().cuda())
            #   B x F x nOutput x maxNeighbor
            # Note that this gather function already reduces the dimension to
            # nOutputNodes.
            # And proceed to compute the maximum along this dimension
            v, _ = torch.max(xNeighbors, dim = 3)
            v = v.unsqueeze(3) # to concatenate with xK
            xK = torch.cat((xK,v),3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # multiply each k-hop max by corresponding weight
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


class MedianLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/27
    """
    MedianLocalActivation creates a localized activation function layer on 
    graphs

    Initialization:

        MedianLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized median activation function layer

    Add graph shift operator:

        MedianLocalActivation.addGSO(GSO) Before applying the filter, we need 
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number 
        of edge features is the same; but the number of nodes can change).
        This function also calculates the 0-,1-,...,K-hop neighborhoods of every
        node

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MedianLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K = 1):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        self.masks = 'None' # no mask yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
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
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(self.S.cpu()), k, outputType='list')
            # compute the k-hop neighborhood
            neighborhood.append(thisNeighborhood)
        self.neighborhood = neighborhood

    def forward(self, x):
        #print(f"Activation Function {np.shape(x)}")
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's
        # It is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        #x = x.unsqueeze(3) # B x F x N x 1
        for k in range(1,self.K+1):
            kHopNeighborhood = self.neighborhood[k-1] 
            # Fetching k-hop neighborhoods of all nodes
            kHopMedian = torch.empty(0)
            # Initializing the vector that will contain the k-hop median for
            # every node
            for n in range(self.N):
                # Iterating over the nodes
                # This step is necessary because here the neighborhoods are
                # lists of lists. It is impossible to pad them and feed them as
                # a matrix, as this would impact the outcome of the median
                # operation
                nodeNeighborhood = torch.tensor(np.array(kHopNeighborhood[n]))
                neighborhoodLen = len(nodeNeighborhood)
                gatherNode = nodeNeighborhood.reshape([1, 1, neighborhoodLen])
                gatherNode = gatherNode.repeat([batchSize, dimNodeSignals, 1])
                # Reshaping the node neighborhood for the gather operation
                xNodeNeighbors = torch.gather(x, 2, gatherNode.long().cuda())
                # Gathering signal values in the node neighborhood
                nodeMedian,_ = torch.median(xNodeNeighbors, dim = 2,
                                            keepdim=True)
                # Computing the median in the neighborhood
                kHopMedian = torch.cat([kHopMedian.cuda(),nodeMedian.cuda()],2)
                # Concatenating k-hop medians node by node
            kHopMedian = kHopMedian.unsqueeze(3) # Extra dimension for
            # concatenation with the previous (k-1)-hop median tensor 
            xK = torch.cat([xK,kHopMedian],3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # Multiplying each k-hop median by corresponding trainable weight
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
        
class NoActivation(nn.Module):
    """
    NoActivation creates an activation layer that does nothing
        It is for completeness, to be able to switch between linear models
        and nonlinear models, without altering the entire architecture model
    Initialization:
        NoActivation()
        Output:
            torch.nn.Module for an empty activation layer
    Forward call:
        y = NoActivation(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes
        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        
        return x
    
    def extra_repr(self):
        reprString = "No Activation Function"
        return reprString

class NoPool(nn.Module):
    """
    This is a pooling layer that actually does no pooling. It has the same input
    structure and methods of MaxPoolLocal() for consistency. Basically, this
    allows us to change from pooling to no pooling without necessarily creating
    a new architecture.
    
    In any case, we're pretty sure this function should never ship, and pooling
    can be avoided directly when defining the architecture.
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, GSO):
        # This is necessary to keep the form of the other pooling strategies
        # within the SelectionGNN framework. But we do not care about any GSO.
        pass

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And do not do anything
        return x

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
                self.nInputNodes, self.nOutputNodes, self.nHops)
        reprString += "no neighborhood needed"
        return reprString




