import sys
import pandas as pd
import numpy as np
from scipy import sparse
import networkx as ntx
import matplotlib.pyplot as plt

import tegrasa as tg

A  = np.loadtxt('circle.adj')
XY = np.loadtxt('circle.xy')
S  = np.loadtxt('circle.dat')

#############################################
# plottings related to the synthetic data set
#############################################

S = np.abs(S+np.random.uniform(low=-0.1,high=0.1,size=(S.shape[0],S.shape[1])))

patterns = tg.tegrasa(Adj=sparse.lil_matrix(A), Signal=S, binary=True)
patterns.run()

scaling = 20
alpha   = 0.7
edgecolor = 'gray'
color_bg  = 'w'

G = ntx.from_numpy_matrix(A)

########
# SIGNAL
########
fig=plt.figure(1)
fig.set_facecolor(color_bg)
ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor,alpha=alpha)
plt.scatter(XY[:,0],XY[:,1],s=8,c=S[:,39],cmap=plt.cm.plasma)
plt.axis('equal')
plt.axis('off')
plt.title('slice 39');

##########################
# EDGE NODES CONFIGURATION
##########################
fig=plt.figure(2)
fig.set_facecolor(color_bg)
ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor, alpha=alpha)
sedges=patterns.get_edgemap(39)
plt.scatter(XY[:,0],XY[:,1], s=10*sedges,c=sedges, cmap=plt.cm.gist_yarg)
plt.axis('equal')
plt.axis('off')
plt.title('slice 39');

#################
# ENTROPY DIAGRAM
#################
fig = plt.figure(3, figsize=[16,3])
cf=patterns.get_clusters();
y=patterns.get_entropy();
x=range(len(y));
plt.plot(y,c='gray');
plt.scatter(x,y,s=scaling, c=cf, cmap='jet');
cur_axes = fig.gca()
cur_axes.axes.get_yaxis().set_visible(False)
plt.xlabel('Time')
plt.tight_layout()

plt.show()
