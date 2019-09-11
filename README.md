# GLoG

The Graph Laplacian of Gaussian (GLoG) is a method to visual analysis of spatio-temporal data. 

The method proposes a novel filter, called GLoG, which is the counterpart for graphs of the so-called Laplacian of Gaussian edge detection method widely used in image processing. The proposed GLoG filter can identify spatial locations of abrupt changes in a signal, uncovering regions (boundaries) where the signal changes its properties. Moreove, we rely on the GLoG filter to define the concept of entropy of time slices, from which we derive a temporal entropy diagram. The latter allows the visual identification of time instances where observed boundaries are likely to happen (lower entropy time instants) as well as moments where observed boundaries are less expected (higher entropy time instants). The resulting analysis makes easier the visual identification of expected and unexpected patterns over time.

The provided ``class GLoG`` code implements the GLoG method described in the work:

L.G. Nonato, F. Petronetto, C.T. Silva. GLoG: Laplacian of Gaussian for Spatial Pattern Detection in Spatio-Temporal Data (https://arxiv.org/abs/1909.03993).

Please cite/acknowledge the work above when using the code in your work.

---
Example: <br>
The following files define synthetic data set with 600 nodes.

__circle.adj__: N x N adjacency matrix where N is the number of nodes.

__circel.xy__: N x 2 position matrix where N is the number of nodes.

__circle.dat__: N x k signal matrix where where N is the number of nodes and k is the number of time slices.

The code

- _synthetic.py_

shows how to use GLoG to create the entropy diagram of the signal.
