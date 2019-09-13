import sys
import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy import sparse

import g_log as glog

#########
# loading
#########
A  = np.loadtxt('circle.adj')
XY = np.loadtxt('circle.xy')
S  = np.loadtxt('circle.dat')
signal = np.abs(S+np.random.uniform(low=-0.1,high=0.1,size=(S.shape[0],S.shape[1])))
adj    = sparse.lil_matrix(A)
sigma  = 1.3 
stdp   = 3.0

####################
# compute edge nodes
####################
n = signal.shape[0]
m = signal.shape[1]
my_glog = glog.g_log(Adj=adj)
edges_ic = my_glog.edge_detection(signal, sigma=sigma, stdp=stdp, binary=True)
sge = edges_ic[1]

#################
# compute entropy
#################
nbins = 2
pdfs = np.zeros((n,nbins))
nt_edges = np.sum(sge,axis=1)
for i in range(n):
     pdfs[i,:] = (1.0/nbins)*np.histogram(sge[i,:], bins=nbins, range=(0.0,1.0), density=True)[0]
entro = np.zeros((m,))
rows = np.arange(n)
for i in range(m):
     truncated = np.trunc(nbins*sge[:,i]).astype(int)
     truncated[np.where(truncated == nbins)] = nbins-1
     pis = pdfs[rows.ravel(),truncated.ravel()]
     entro[i] = -np.sum(pis*np.log(pis))/n

###################################
# cluster edge nodes configurations
###################################
clustering = KMeans(n_clusters=3, random_state=0).fit(sge.T)
clusters = clustering.labels_

#############################################
# plotting related to the synthetic data set
#############################################
scaling = 20
alpha   = 0.7
edgecolor = 'gray'
color_bg  = 'w'

#######
# graph
#######
G = ntx.from_numpy_matrix(A)

########
# SIGNAL
########
fig=plt.figure(1)
fig.set_facecolor(color_bg)
ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor, alpha=alpha)
plt.scatter(XY[:,0],XY[:,1], s=8, c=S[:,39], cmap=plt.cm.plasma)
plt.axis('equal')
plt.axis('off')
plt.title('slice 39');

##########################
# EDGE NODES CONFIGURATION
##########################
fig=plt.figure(2)
fig.set_facecolor(color_bg)
ntx.draw_networkx_edges(G, pos=XY, edge_color=edgecolor, alpha=alpha)
sedges=sge[:,39]
plt.scatter(XY[:,0],XY[:,1], s=10*sedges, c=sedges, cmap=plt.cm.gist_yarg)
plt.axis('equal')
plt.axis('off')
plt.title('slice 39');

#################
# ENTROPY DIAGRAM
#################
fig = plt.figure(3, figsize=[16,3])
y=entro;
x=range(len(y));
plt.plot(y,c='gray');
plt.scatter(x,y,s=scaling, c= clusters, cmap='jet');
cur_axes = fig.gca()
cur_axes.axes.get_yaxis().set_visible(False)
plt.xlabel('Time')
plt.tight_layout()

plt.show()
