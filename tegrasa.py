# Author: Luis Gustavo Nonato  -- <gnonato@icmc.usp.br>

import sys
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from scipy import stats

import g_log as glog

def check_data_type(dt):
    try:
        if type(dt) is not np.ndarray:
            if sparse.issparse(dt) is False:
                raise TypeError()

    except TypeError:
        print('----- tegrasa Error -----')
        print('Adj must be a "numpy array" or "sparse matrix"')
        sys.exit()

def check_dimensions(dadj,ds):
    try:
        if dadj[0] != ds[0]:
            raise ValueError()

    except ValueError:
        print('----- tegrasa Error -----')
        print('Number of rows in Adj is different from the number of rows in Signal')
        sys.exit()

class tegrasa():
    def __init__(self, Adj = None, Signal = None, sigma=3.0, std = 3.0, binary = False):
        ''' Adj:    adjacency matrix representing graph edges.
                    Adj must be a scipy sparce matrix or a numpy ndarray.
            Signal: a 2D numpy array where rows correspoind to nodes
                    and columns to the signal defined on the nodes.
                    The index of the node is the same as the adjacency matrix. For instance,
                    the first rows corresponds to the time series signal associated to node 0
        '''

        try:
            if Adj is None:
                raise ValueError('Adjacency Matrix Not Provided !!')

            if Signal is None:
                raise ValueError('Signal Matrix Not Provided !!')

        except ValueError as v:
            print('----- tegrasa Error -----')
            print(v)
            sys.exit()

        else:
            check_data_type(Adj)
            if Adj is not sparse.lil.lil_matrix:
                self.adj = sparse.lil_matrix(Adj)
            else:
                self.adj = Adj

            check_dimensions(Adj.shape,Signal.shape)
            if Signal.shape[1] is None:
                Signal = Signal.reshape(-1,1)

            self.signal = Signal
            self.sigma = sigma
            self.std = std
            self.binary = binary

    def smoothed_graph_edges(self):
        ''' Compute graph edges in each time slice and smooth them out '''

        n = self.signal.shape[0]
        m = self.signal.shape[1]
        self.sge = np.zeros((n,m))

        edges = glog.g_log(Adj = self.adj)

        ###### Synthetic Circle #######
        edges_ic = edges.edge_detection(self.signal,sigma=1.2,stdp=3.0,binary=True)
        self.edges = edges_ic[1];
        if (self.binary == False):
            #self.sge = np.abs(edges.smooth(self.sge,smooth_type='GAUSS',sigma = 2.0, normalize = True))
            self.sge = edges.smooth(self.edges,smooth_type='GAUSS',sigma = 2.0, normalize = True)
            self.sge[np.where(self.sge < 0.0)] = 0.0
        else:
            self.sge = self.edges

    def edge_clusters(self,k=4):
        edges = glog.g_log(Adj = self.adj)
        self.sges = edges.smooth(self.sge,smooth_type='GAUSS',sigma = 1.5, normalize = True)
        clustering = KMeans(n_clusters=k, random_state=0).fit(self.sge.T)
        self.clusters = clustering.labels_


    def edge_variation(self):
        ''' Compute how much variation of edges in each time slice '''
        n = self.signal.shape[0]
        m = self.signal.shape[1]
        self.ev = np.zeros((m,))

        self.ev[0] = np.sum(np.abs(self.sge[:,0] - self.sge[:,1]))/n
        self.ev[m-1] = np.sum(np.abs(self.sge[:,m-1] - self.sge[:,m-2]))/n
        for i in range(1,m-1):
            self.ev[i] = (np.sum(np.abs(self.sge[:,i] - self.sge[:,i+1]))/n + np.sum(np.abs(self.sge[:,i] - self.sge[:,i-1]))/n)/2

    def edge_entropy(self):
        ''' Compute the entropy of each time slice '''
        n = self.signal.shape[0]
        m = self.signal.shape[1]
        if self.binary == False:
            nbins = 15
        else:
            nbins = 2
        pdfs = np.zeros((n,nbins))
        nt_edges = np.sum(self.sge,axis=1)

        nbins = 2
        for i in range(n):
            pdfs[i,:] = (1.0/nbins)*np.histogram(self.sge[i,:],bins=nbins,range=(0.0,1.0),density=True)[0]

        self.entro = np.zeros((m,))
        rows = np.arange(n)
        eps = 1.0e-10
        for i in range(m):
            truncated = np.trunc(nbins*self.sge[:,i]).astype(int)
            truncated[np.where(truncated == nbins)] = nbins-1
            pis = pdfs[rows.ravel(),truncated.ravel()]
            self.entro[i] = -np.sum(pis*np.log(pis))/n
            self.pdfs = pdfs

    def run(self):
        self.smoothed_graph_edges()
        self.edge_entropy()
        self.edge_clusters(k=4)


    def get_edges(self,k):
        return(self.edges[:,k])

    def get_edgemap(self,k):
        return(self.sge[:,k])

    def get_edgevariation(self):
        return(self.ev)

    def get_entropy(self):
        return(self.entro)

    def get_pdfs(self):
        return(self.pdfs)

    def get_clusters(self):
        return(self.clusters)
