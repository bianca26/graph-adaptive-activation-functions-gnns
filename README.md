# Graph-Adaptive Activation Functions for Graph Neural Networks
This is a PyTorch implementation of graph-adaptive activation functions for Graph Neural Networks (GNNs). For any questions or suggestions, please e-mail Bianca Iancu at <bianca.iancu026@gmail.com>.

When using part of this code, please cite the following paper

Bianca Iancu, Luana Ruiz, Alejandro Ribeiro, and Elvin Isufi, "Graph-Adaptive Activation Functions for Graph Neural Networks". *IEEE International Workshop on MACHINE LEARNING FOR SIGNAL PROCESSING (MLSP 2020)*, IEEE, September 21-24, 2020.

Other paper on GNNs by the authors is

Bianca Iancu, and Elvin Isufi, "Towards Finite-Time Consensus with Graph Convolutional Neural Networks". *28th European Signal Processing Conference (EUSIPCO 2020)*, IEEE, January 18-22, 2021.

1. [Introduction](#intro)
2. [Graph-Adaptive Activation Functions](#ga)
3. [Code](#code)

<a name="intro"></a>
## 1. Introduction

We consider data represented on a graph with N nodes and M edges. On the vertices of the graph, we define a graph signal **x** whose *i*th component is the value at node *i* in the graph. We consider applications where graph signals are processed in a *distributed* fashion. A typical example is in sensor networks without access to a centralized processing unit and where each sensor communicates only with its neighbor sensors. 

Associated to the graph is the shift operator (GSO) matrix **S**, whose sparsity pattern matches the graph structure. That is, **S**  has non-zero values only for the entries associated to edges in the graph. Commonly used GSOs include the adjacency matrix, the graph Laplacian, and their normalized and translated forms. The choice for the GSO varies depending on the application, and different choices have different trade-offs. For our experiments, we employ the adjacency matrix as the GSO. The main operation carried out by the GSO is the shifting of a graph signal **x** over the graph, that is

![shift](images/shift.png =100x20)

<img src="images/shift.png" width="300" height="80" />

<a name="ga"></a>
## 2. Graph-Adaptive Activation Functions

<a name="code"></a>
## 3. Code

