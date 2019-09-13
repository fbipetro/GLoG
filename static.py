import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as ntx

from scipy import sparse

import g_log as glog

n=30

# creating edges
el = [[i,i+1] for j in range(0,n) for i in range(n*j,n*(j+1)-1)]
el.extend([[i,i+n] for j in range(0,n-1) for i in range(n*j,n*(j+1))])

# adjacent matrix
A = sparse.lil.lil_matrix((n*n,n*n))
for e in el:
    A[e[0],e[1]] = 1
    A[e[1],e[0]] = 1

# creating a step function
fs = np.ones((n*n,1))
fs[n*n//2:,0] = -1.0
fs = fs + np.random.uniform(low=-0.3, high=0.3, size=(n*n,1))

# creating a spike function
nodes = np.random.randint(low = 0, high = n*n-1, size=5)
fspk = np.zeros((n*n,1))
fspk[nodes] = 1.0

# for rendering purpose only
X = np.asarray([[i,j] for j in range(0,n) for i in range(0,n)],dtype = float)
X[:,0] = X[:,0]/n
X[:,1] = X[:,1]/n
G = ntx.from_numpy_matrix(A.toarray())


my_glog = glog.g_log(Adj = A)

#######################################
######   Smoothing the Spike   ########
#######################################

###### smoothing a spike with smooth_cheby ########
fspk_smoothed = my_glog.smooth(fspk, smooth_type = 'ARMA', plambda = 5)


fig1,axs = plt.subplots(1,2,sharex = True,sharey=True)
ax = axs[0]
ax.set_title('graph spike signal')
ax.set_facecolor("white")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ntx.draw_networkx_edges(G, pos=X, edge_color='gray', alpha=0.5, ax=ax)
sctt=ax.scatter(X[:,0],X[:,1], s=10*fspk[:,0], c=fspk[:,0], cmap=plt.cm.plasma)
fig1.colorbar(sctt,ax=ax)

ax = axs[1]
ax.set_facecolor("white")
ax.set_title('smoothing spikes')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ntx.draw_networkx_edges(G, pos=X, edge_color='gray', alpha=0.5, ax=ax)
sctt=plt.scatter(X[:,0],X[:,1], s=10*fspk_smoothed[:,0], c=fspk_smoothed[:,0], cmap=plt.cm.plasma)
fig1.colorbar(sctt,ax=ax)



#######################################
######        G_LoG Step         ######
#######################################

######### edge nodes detection ##########
edges_ic = my_glog.edge_detection(fs,sigma=3,stdp=3.0,binary=True)
#sedges = edges_ic[0] #without treshold (stdp user parameter)
sedges = edges_ic[1] #after treshold

fig2,axs = plt.subplots(1,2,sharex = True,sharey=True)
ax = axs[0]
ax.set_title('graph signal')
ax.set_facecolor("white")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ntx.draw_networkx_edges(G, pos=X, edge_color='gray', alpha=0.5, ax=ax)
sctt=ax.scatter(X[:,0],X[:,1], s=10, c=fs[:,0], cmap=plt.cm.plasma)
fig2.colorbar(sctt,ax=ax)

ax = axs[1]
ax.set_facecolor("white")
ax.set_title('edge nodes')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ntx.draw_networkx_edges(G, pos=X, edge_color='gray', alpha=0.5, ax=ax)
sctt=plt.scatter(X[:,0],X[:,1], s=10*sedges, c=sedges[:,0], cmap=plt.cm.gist_yarg)
fig2.colorbar(sctt,ax=ax)

plt.show()
