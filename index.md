## <center>Multiview Graph Convolutional Networks with Attention Mechanism<center>
This repository contains the author's implementation in Tensorflow for the paper "Multiview Graph Convolutional Networks with Attention Mechanism".


## <center>Abstract<center>
Recent advances in graph convolutional networks (GCNs), mainly focusing on how to exploit the information from different hops of neighbors in an efficient way, have brought substantial improvement on many graph data modelling tasks. Most of the existing GCN-based models however are built on the basis of a fixed adjacency matrix, i.e., a single view topology of the underlying graph. That inherently limits the expressive power of the developed models when the given adjacency matrix that is viewed as an approximation of the unknown graph topology does not fully reflect the `ideal' structure knowledge. In this paper, we propose a novel framework, termed Multiview Graph Convolutional Networks with Attention Mechanism (MAGCN), by incorporating multiple views of topology and attention based feature aggregation strategy into the computation of graph convolution. Furthermore, we present some theoretical analysis about the expressive power and flexibility of MAGCN, which provides a general explanation on why multi-view based methods can potentially outperform the ones relying on a single view. Our experimental study demonstrates the state-of-the-art accuracies of MAGCN on Cora, Citeseer, and Pubmed datasets. Robustness analysis is also given to show the advantage of MAGCN in handling some uncertainty issues in node classification tasks.


## Overview
>The overall structure of MAGCN
![The structures of MAGCN](https://github.com/ICML2020-submission/MAGCN/blob/master/images/MAGCN_structure.jpg)


## Visualization
>t-SNE visualization for the computed feature representations of a pre-trained model’s first hidden layer on the Cora dataset:
GCN (left) and our MAGCN (right). Node colors denote classes.
![t-SNE visualization.](https://github.com/ICML2020-submission/MAGCN/blob/master/images/visualization.jpg)

## Dependencies

-Python (>=3.5)

-Tensorflow (>=1.12.0)

-Keras (>=2.0.9)
