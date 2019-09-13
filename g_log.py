import sys
import numpy as np
from scipy import sparse
from scipy import integrate

from functools import reduce

global pi
pi = 3.1415

def ARMA(t,a,m,plambda):
    x = a*(np.cos(t)+1)
    g = np.cos(m*t)*1.0/(1.0+plambda*x)
    return(g)

def FGauss(t,a,m,sigma):
    x = a*(np.cos(t)+1)
    g = np.cos(m*t)*(-4.0*pi**2)*(x**2)*np.exp(-(sigma**2)*(x**2)/2.0)
    return(g)

def Gauss(t,a,m,sigma):
    x = a*(np.cos(t)+1)
    g = np.cos(m*t)*np.exp(-(sigma**2)*(x**2)/2.0)
    return(g)

def check_data_type(dt):
    try:
        if type(dt) is not np.ndarray:
            if type(dt) is not sparse.lil.lil_matrix:
                raise TypeError()

    except TypeError:
        print('----- G_LoG Error -----')
        print('Xdata must be a "numpy array" or "sparse lil matrix"')
        sys.exit()

###########################
######     G_LoG     ######
###########################

class g_log():
    ''' Class to compute LoG edges or smooth out functions '''

    def __init__(self, Adj = None):
       ''' The adjacency matrix Adj must be an scipy lil_matrix '''

       if Adj is not None:
           check_data_type(Adj)

       self.adj = Adj

#######
    def set_graph(self,Adj = None):
        check_data_type(Adj)
        self.adj = Adj

#######
    def edge_detection(self, Xdata, sigma=2.0, stdp=0.0, binary=False, cheby_degree=6):
        check_data_type(Xdata)

        try:
            if self.adj is None:
                raise ValueError()
        except ValueError:
            print('----- g_log.edge_detection Error -----')
            print("Graph dot defined... see G_LoG.set_graph()")
            sys.exit()

        log = self.LoG(Xdata, sigma=sigma, cheby_degree=cheby_degree)

        # getting graph edges
        A = self.adj
        edges_ids = sparse.find(A > 1.0e-7)
        # number of edges
        ll=[]
        lr=[]
        for i in range(edges_ids[0].size):
            if edges_ids[0][i] < edges_ids[1][i]:
                ll.append(edges_ids[0][i])
                lr.append(edges_ids[1][i])

        edges_ids = [ll,lr]
        n_edges = len(edges_ids[0])

        ###### finding the more intense zero crossings #####
        n = A.shape[0]
        conv = lambda i,j: i*n+j

        # store the crossing value of each edge
        # indexed based on the order of edges_ids
        zcross_intensity = np.zeros((n_edges,Xdata.shape[1]))

        # computing the zero-crossing intensity in each edge
        cross_sign = log[edges_ids[0]]*log[edges_ids[1]]
        cross_ids = np.where(cross_sign < 0.0)
        self.edges_ids = [[edges_ids[0][i] for i in cross_ids[0]],[edges_ids[1][i] for i in cross_ids[0]]]
        cross_dif = np.abs(log[edges_ids[0]] - log[edges_ids[1]])
        zcross_intensity[cross_ids] = cross_dif[cross_ids]

        zcross_intensity_mean = np.mean(zcross_intensity,axis=0)
        zcross_intensity_std = np.std(zcross_intensity,axis=0)

        # extract the most intense zero-crossing
        zcross_ids = np.where(zcross_intensity >= zcross_intensity_mean + stdp*zcross_intensity_std)
        idxr = [edges_ids[0][i] for i in zcross_ids[0]]
        idxc = [edges_ids[1][i] for i in zcross_ids[1]]

        # nodes where the zero crossing is more intense
        edge_nodes_i = np.zeros((n,Xdata.shape[1]))
        edge_nodes_c = np.zeros((n,Xdata.shape[1]))

        if binary == False:
            for i in range(zcross_ids[0].shape[0]):
                edge_nodes[edges_ids[0][zcross_ids[0][i]],zcross_ids[1][i]] = np.abs(log[edges_ids[0][zcross_ids[0][i]],zcross_ids[1][i]])
                edge_nodes[edges_ids[1][zcross_ids[0][i]],zcross_ids[1][i]] = np.abs(log[edges_ids[1][zcross_ids[0][i]],zcross_ids[1][i]])
        else:
            for i in range(cross_ids[0].shape[0]):
                edge_nodes_c[edges_ids[0][cross_ids[0][i]],cross_ids[1][i]] = 1
                edge_nodes_c[edges_ids[1][cross_ids[0][i]],cross_ids[1][i]] = 1
            for i in range(zcross_ids[0].shape[0]):
                edge_nodes_i[edges_ids[0][zcross_ids[0][i]],zcross_ids[1][i]] = 1
                edge_nodes_i[edges_ids[1][zcross_ids[0][i]],zcross_ids[1][i]] = 1

        return edge_nodes_c, edge_nodes_i

#######
    def cheby_approx(self,Xdata,L,c,a,degree):
        CT = (1.0/a)*(L - a*sparse.lil_matrix(np.identity(L.shape[0])))

        Xdata = sparse.lil_matrix(Xdata)
        Tm_2 = Xdata
        Tm_1 = np.dot(CT,Xdata)
        s = (0.5*c[0])*Xdata + c[1]*Tm_1
        for m in range(2,degree):
            k1 = np.dot(2.0*CT,Tm_1)
            Tm = np.subtract(k1,Tm_2)
            s = s + c[m]*Tm
            Tm_2 = Tm_1
            Tm_1 = Tm

        return(s)

#######
    def smooth(self,Xdata, smooth_type='ARMA', normalize = True, plambda = 2.0, sigma = 2.0, cheby_degree = 6):
        ''' Xdata:
                function (or set of functions, one per column of Xdata) defined on the nodes of the graph
            smooth_type:
                'ARMA': use the ARMA function 1/(1+lambda*x) as kernel
                'GAUSS': use the Gauss function exp(-sigma^2*x^2/2) as kernel
                       - sigma must be greater than one
                       - if 'ARMA' is chosen sigma has no effect
                       - if 'GAUSS' is chosen plambda has no effect
            normalize:
                smooth Xdata and normalize the result such that the original scale is preserved
            cheby_degree:
                the degree of the chebychev polynomials used in the approximation
        '''
        check_data_type(Xdata)

        try:
            if self.adj is None:
                raise ValueError()
        except ValueError:
            print('----- g_log.smooth Error -----')
            print("Graph dot defined... see G_LoG.set_graph()")
            sys.exit()


        cheby_degree = cheby_degree + 1

        if self.adj is not sparse.lil.lil_matrix:
            A = sparse.lil_matrix(self.adj)
        else:
            A = self.adj

        D = sparse.lil_matrix(np.diag(np.ravel(np.sum(A,axis=0))))
        L = D - A 
        lmax = sparse.linalg.eigsh(L,k=1)[0]
        self.lmax = lmax

        if smooth_type == 'ARMA':
            spec_filter = ARMA
            param = plambda
        else:
            if smooth_type == 'GAUSS':
                spec_filter = Gauss
                param = sigma
            else:
                print("Smooth Function "+smooth_type+" not known !!")
                sys.exit()

        a = lmax[0]/2.0
        c = np.zeros((cheby_degree,))
        for m in range(cheby_degree):
            c[m]=2.0/pi*integrate.quadrature(spec_filter,0,pi,args=(a,m,param))[0]

        smooth_data = self.cheby_approx(Xdata,L,c,a,cheby_degree).toarray()

        if normalize is True:
            t1 = np.max(np.abs(smooth_data),axis=0)
            t2 = np.divide(smooth_data,t1)
            t3 = np.max(np.abs(Xdata),axis=0)
            smooth_data = np.multiply(np.broadcast_to(t3,(smooth_data.shape[0],smooth_data.shape[1])),t2)

        return(smooth_data)

#######
    def LoG(self,Xdata, sigma = 2.0, cheby_degree = 6):
        try:
            if self.adj is None:
                raise ValueError()
        except ValueError:
            print('----- g_log.LoG Error -----')
            print("Graph dot defined... see g_log.set_graph()")
            sys.exit()

        check_data_type(Xdata)

        cheby_degree = cheby_degree + 1

        if self.adj is not sparse.lil.lil_matrix:
            A = sparse.lil_matrix(self.adj)
        else:
            A = self.adj

        ######## Computing LoG edge detection ##########
        D = sparse.lil_matrix(np.diag(np.ravel(np.sum(A,axis=0))))
        L = D - A
        lmax = sparse.linalg.eigsh(L,k=1)[0]
        self.lmax = lmax

        a = lmax[0]/2.0
        c = np.zeros((cheby_degree,))
        for m in range(cheby_degree):
            c[m]=2.0/pi*integrate.quadrature(FGauss,0,pi,args=(a,m,sigma))[0]

        log = self.cheby_approx(Xdata,L,c,a,cheby_degree).toarray()

        return(log)
