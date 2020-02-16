<h1 align = "center">Multi-View Graph Convolutional Networks with Attention Mechanism</h1>

<h1 align = "center">Abstract</h1> 

Recent advances in graph convolutional networks (GCNs), mainly focusing on how to exploit the information from different hops of neighbors in an efficient way, have brought substantial improvement on many graph data modelling tasks. Most of the existing GCN-based models however are built on the basis of a fixed adjacency matrix, i.e., a single view topology of the underlying graph. That inherently limits the expressive power of the developed models when the given adjacency matrix that is viewed as an approximation of the unknown graph topology does not fully reflect the `ideal' structure knowledge. In this paper, we propose a novel framework, termed Multiview Graph Convolutional Networks with Attention Mechanism (MAGCN), by incorporating multiple views of topology and attention based feature aggregation strategy into the computation of graph convolution. Furthermore, we present some theoretical analysis about the expressive power and flexibility of MAGCN, which provides a general explanation on why multi-view based methods can potentially outperform the ones relying on a single view. Our experimental study demonstrates the state-of-the-art accuracies of MAGCN on Cora, Citeseer, and Pubmed datasets. Robustness analysis is also given to show the advantage of MAGCN in handling some uncertainty issues in node classification tasks.

<h1 align = "center">Motivation</h1>

Despite that GCN and its variants/extensions have shown their great success on node classification tasks, almost all of these models are developed based on a fixed adjacency matrix given in advance, in other words, a single view graph topology. Inherently, the expressive power of the resulted model may be limited due to the potential information discrepancy between the adjacency matrix and the (unknown) target one. As such, it is logical to consider two practical questions: 

Q1: Is the given topology (adjacency matrix) trustable?

Q2: How to carry out the neighborhood aggregation or message passing when multi-view topologies of the graph are provided?

In this paper, we propose a novel framework, termed Multiview Graph Convolutional Networks with Attention Mechanism (MAGCN), by incorporating multiple views of topology and attention based feature aggregation strategy into the computation of graph convolution.

<h1 align = "center">Overview</h1>

<div align="center">
    <img src="images/MAGCN_structure.jpg" width="80%" height ="80%" alt="MAGCN_structure.jpg" />
</div>
<p align = 'center'>
    <small>The overall structure of MAGCN.</small>
</p>
