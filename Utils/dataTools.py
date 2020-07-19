# 2018/12/4~~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu

"""
Data management module
The following methods and classes are copied from the code by Fernando Gama, available at
https://github.com/alelab-upenn/graph-neural-networks
    - normalizeData method
    - changeDataType method
    - _data class
    - _dataForClassification class


Tools to manage data

SourceLocalization (class): creates the datasets for a source localization 
    problem
FiniteTimeConsensus (class): created the dataset for the finite-time consensus problem
    problem
"""



import os
import pickle
import hdf5storage # This is required to import old Matlab(R) files.
import urllib.request # To download from the internet
import zipfile # To handle zip files
import gzip # To handle gz files
import shutil # Command line utilities

import numpy as np
import torch

import Utils.graphTools as graph

zeroTolerance = 1e-9 # Values below this number are considered zero.


def normalizeData(x, ax):
    """
    normalizeData(x, ax): normalize data x (subtract mean and divide by standard
    deviation) along the specified axis ax
    """

    thisShape = x.shape  # get the shape
    assert ax < len(thisShape)  # check that the axis that we want to normalize
    # is there
    dataType = type(x)  # get data type so that we don't have to convert

    if 'numpy' in repr(dataType):

        # Compute the statistics
        xMean = np.mean(x, axis=ax)
        xDev = np.std(x, axis=ax)
        # Add back the dimension we just took out
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)

    elif 'torch' in repr(dataType):

        # Compute the statistics
        xMean = torch.mean(x, dim=ax)
        xDev = torch.std(x, dim=ax)
        # Add back the dimension we just took out
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - xMean) / xDev

    return x


def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """

    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.

    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.

    # If we can't recognize the type, we just make everything numpy.

    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype

    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype=dataType)

    # This only converts between numpy and torch. Any other thing is ignored
    return x

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None

        self.labelIDs = {}
        self.labelIDs['train'] = None
        self.labelIDs['valid'] = None
        self.labelIDs['test'] = None
        
    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        z = self.labelIDs[samplesType]

        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
                z = z[selectedIndices]

            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]
                z = z[args[0]]


        return x, y, z

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers. To do this we need to
        # match the desired dataType to its int counterpart. Typical examples
        # are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        labelType = str(self.samples['train']['labels'].dtype)
        if 'int' in labelType:
            if 'numpy' in repr(dataType) or 'np' in repr(dataType):
                if '64' in labelType:
                    labelType = np.int64
                elif '32' in labelType:
                    labelType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in labelType:
                    labelType = torch.int64
                elif '32' in labelType:
                    labelType = torch.int32
        else: # If there is no int, just stick with the given dataType
            labelType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        if 'torch' in repr(dataType): # If it is torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                       torch.tensor(self.samples[key]['signals']).type(dataType)
                self.samples[key]['labels'] = \
                       torch.tensor(self.samples[key]['labels']).type(labelType)
        else: # If it is not torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                                          dataType(self.samples[key]['signals'])
                self.samples[key]['labels'] = \
                                          labelType(self.samples[key]['labels'])

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are correctly labels) so justifies the use of
    # another internal class.
    
    def __init__(self):
        
        super().__init__()
    

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """

        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat.type(torch.int64) - y.type(torch.int64)) > tol)
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy
        

class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem
    We already give the signals split in train, validation and test sets, with their associated labels and label IDs.
    We use this class for defining the data in a format that allows using the methods of the 'data' class. -> Notice it
    inherits the functionality of the 'dataForClassification' class, which inherits it from the 'data' class.

    """

    def __init__(self, data_train, labels_train, labelIDs_train,
                 data_validation, labels_validation, labelIDs_validation,
                 data_test, labels_test, labelIDs_test, dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = len(data_train)
        self.nValid = len(data_validation)
        self.nTest = len(data_test)


        self.samples['train']['signals'] = data_train
        self.samples['train']['labels'] = labels_train
        self.labelIDs['train'] = labelIDs_train


        self.samples['valid']['signals'] = data_validation
        self.samples['valid']['labels'] = labels_validation
        self.labelIDs['valid'] = labelIDs_validation


        self.samples['test']['signals'] = data_test
        self.samples['test']['labels'] = labels_test
        self.labelIDs['test'] = labelIDs_test


        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    

class Finite_time_consensus(_dataForClassification):
    """
    Finite_time_consensus: class for generating the data for the finite time consensus problem.
    We already give the signals split in train, validation and test sets, with their associated labels and label IDs.
    We use this class for defining the data in a format that allows using the methods of the 'data' class. -> Notice it
    inherits the functionality of the 'dataForClassification' class, which inherits it from the 'data' class.
    """


    def __init__(self, data_train, labels_train, labelIDs_train,
                 data_validation, labels_validation, labelIDs_validation,
                 data_test, labels_test, labelIDs_test,
                 dataType=np.float64, device='cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.labelID = None
        self.dataType = dataType
        self.device = device
        self.nTrain = len(data_train)
        self.nValid = len(data_validation)
        self.nTest = len(data_test)


        self.samples['train']['signals'] = data_train
        self.samples['train']['labels'] = labels_train
        self.labelIDs['train'] = labelIDs_train

        self.samples['valid']['signals'] = data_validation
        self.samples['valid']['labels'] = labels_validation
        self.labelIDs['valid'] = labelIDs_validation


        self.samples['test']['signals'] = data_test
        self.samples['test']['labels'] = labels_test
        self.labelIDs['test'] = labelIDs_test


        self.astype(self.dataType)
        self.to(self.device)

    def evaluate(self, yHat, y):
        # y and yHat should be of the same dimension, where dimension 0 is the
        # number of samples
        N = y.shape[0]  # number of samples
        assert yHat.shape[0] == N
        # And now, get rid of any extra '1' dimension that might appear
        # involuntarily from some vectorization issues.
        y = y.squeeze()
        yHat = yHat.squeeze()
        # Yet, if there was only one sample, then the sample dimension was
        # also get rid of during the squeeze, so we need to add it back
        if N == 1:
            y = y.unsqueeze(0)
            yHat = yHat.unsqueeze(0)

        # Now, we compute the RMS
        if 'torch' in repr(self.dataType):
            mse = torch.nn.functional.mse_loss(yHat, y)
            rmse = torch.sqrt(mse)
        else:
            mse = np.mean((yHat - y) ** 2)
            rmse = np.sqrt(mse)

        return rmse


        

