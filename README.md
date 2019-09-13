# GLoG

The Graph Laplacian of Gaussian (GLoG) is a filter defined for graph signals.

The proposed GLoG filter can be used to:

1. identify spatial locations of abrupt changes in a graph signal, uncovering regions (more precisely edges) where the signal changes its properties. See Example 1.

2. define the concept of entropy of time slices for a time-varying data, of which a temporal entropy diagram is defined. The entropy diagram can be use to visual identification of time instances where observed boundaries are likely to happen (lower entropy time instants) as well as moments where observed boundaries are less expected (higher entropy time instants). See Example 2.

The provided ``class GLoG`` code implements the GLoG filter described in the work:

L.G. Nonato, F. Petronetto, C.T. Silva. GLoG: Laplacian of Gaussian for Spatial Pattern Detection in Spatio-Temporal Data (https://arxiv.org/abs/1909.03993).

Please cite/acknowledge the work above when using the code in your work.

---
Example 1: <br>

The code

- _static.py_

show how to use GLoG filter to define edge nodes configuration of a static signal. 

---
Example 2: <br>
Following files define synthetic data set with N=600 nodes and k=100 time slices.

__circle.adj__: N x N adjacency matrix where N is the number of nodes.

__circel.xy__: N x 2 position matrix where N is the number of nodes.

__circle.dat__: N x k signal matrix where where N is the number of nodes and k is the number of time slices.

The code

- _temporal.py_

show how to use GLoG filter to create the entropy diagram of the time-varying signal. 
