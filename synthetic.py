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
alpha = 0.7
edgecolor = 'gray'
color_bg = 'w'

G = ntx.from_numpy_matrix(A)
slices = [5,12,30]

fig1=plt.figure(1)
fig1.set_facecolor(color_bg)
for i in range(3):
	ax1 = fig1.add_subplot(1,3,i+1)
	ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor,alpha=alpha)
	plt.scatter(XY[:,0],XY[:,1],s=8,c=S[:,slices[i]],cmap=plt.cm.plasma)
	plt.axis('equal')
	plt.axis('off')

fig2=plt.figure(2)
fig2.set_facecolor(color_bg)
for i in range(3):
	ax2 = fig2.add_subplot(1,3,i+1)
	ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor,alpha=alpha)
	sedges=patterns.get_edgemap(slices[i])
	plt.scatter(XY[:,0],XY[:,1],s=10*sedges,c=sedges,cmap=plt.cm.gist_yarg)
	plt.axis('equal')
	plt.axis('off')

plt.show()
