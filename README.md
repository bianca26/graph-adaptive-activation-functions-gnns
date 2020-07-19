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

<img src="images/shift.png" width="130" height="60" />

where **x**<sup>(1)</sup> is the signal shifted once by the GSO **S**. Specifically, the shifting operation results in neighboring nodes exchanging information and each node aggregating the incoming information from its one-hop neighbors. An essential property of the GSO is its readily distributed implementation, as the shifting operation only employs information that is locally available at each node.

Based on the GSO **S**, we can further define graph convolutions. These are graph filters that can be written as a polynomial of the GSO **S** as

<img src="images/graph_convolution.png" width="130" height="60" />

where **h** = \[h<sub>0</sub>, ..., h<sub>K</sub>\]<sup>T</sup> is a vector of coefficients. Since **S** has a local implementation, graph filters are local as well. Thus, they can be run distributively.

By employing graph convolutions, we introduce next Graph Convolutional Neural Networks (GCNNs). These are composed of L convolutional layers. Each convolutional layer consists of two fundamental elements: a linear and a nonlinear component. The linear component comprises a collection of graph filters to perform graph convolutions. Specifically, at layer *l*, the GCNN takes as input F<sub>l-1</sub> features {**x**<sub>l-1</sub><sup>g</sup>}<sub>g=1</sub><sup>F<sub>l-1</sub></sup> from layer *l-1* and produces F<sub>l</sub> output features {**x**<sub>l</sub><sup>f</sup>}<sub>f=1</sub><sup>F<sub>l</sub></sup>. Each input feature **x**<sub>l-1</sub><sup>g</sup> is processed by a parallel bank of F<sub>l</sub> graph filters **H**<sub>l</sub><sup>fg</sup>(**S**). The filter outputs are aggregated over the input index *g* to yield the *f*th convolved feature as 

<img src="images/gcnn_convolved_feature.png" />

<a name="ga"></a>
## 2. Graph-Adaptive Activation Functions

<a name="code"></a>
## 3. Code

