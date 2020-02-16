## Multiview Graph Convolutional Networks with Attention Mechanism

This repository contains the author's implementation in Tensorflow for the paper "Multiview Graph Convolutional Networks with Attention Mechanism".


## Overview

> The structures of MAGCN  

<div align="center">
    <img src="docs/images/MAGCN_structure.jpg" width="100%" height ="100%" alt="MAGCN_structure.jpg" />
</div>
<p align = 'center'>
    <small>The overall structure of MAGCN.</small>
</p>

> The visualization results

<div align="center">
    <img src="docs/images/visualization.jpg" width="100%" height ="100%" alt="visualization.jpg" />
</div>
<p align = 'center'>
<small> t-SNE visualization for the computed feature representations of a pre-trained model's first hidden layer on the Cora dataset: GCN (left) and our MAGCN (right). Node colors denote classes. </small>
</p>


## Dependencies

- Python (>=3.5)

- Tensorflow (>=1.12.0)

- Keras (>=2.0.9)

## Implementation

Here we provide the implementation of a MAGCN layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:

 - `data/` contains the necessary dataset files for Cora;
 - `models.py` contains the implementation of the `MAGCN(Model)`;
 - `layers.py` contains the implementation of the `MultiGraphConvolution(Layer)`;
 
 Finally, `train.py` puts all of the above together and may be used to execute a full training run on Cora.
