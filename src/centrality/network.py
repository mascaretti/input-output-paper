import networkx as nx
import numpy as np

def page_rank(input_filepath, output_filepath):
    '''This function takes as input
    a (N x N) IoT, compute the PageRank centrality
    of the graph induced by it and returns its
    PageRank Centrality

    ---------------
    INPUT PARAMETERS:
    A: a (N x N) Numpy array
    '''
    try:
        assert type(input_filepath) is np.ndarray
        A = input_filepath
    except AssertionError:
        A = np.load(input_filepath)

    # Checking if the matrix is square
    try:
        assert A.shape[0] == A.shape[1]
    except:
        n = np.int(np.sqrt(A.shape[0]))
        A = A.reshape(n, n)
    
    # Inducing graph
    G = nx.from_numpy_array(A)

    # PageRank
    pr = nx.pagerank_numpy(G)
    prv = np.fromiter(pr.values(), dtype=np.float64) # from dict to Numpy vector

    # Returning
    np.save(output_filepath, prv)
    return prv


def _mfpt(A):
    '''
    :param A: a Numpy array of dimension (N x N) describing an Input/Output Tables.

    :returns H: a Numpy array of dimension (N x N) describing the Mean First Passage Times.

    Computation of the mean first passage time matrix H of a grah with
    adjacency matrix A using Sherman Morrison
    Note that H(i,j) is MFPT from i to j.
    
    Porting to Python from the Matlab Script of
    Author: Florian Bloechl, July 2009 
    http://cmb.helmholtz-muenchen.de
    This software is for non-commercial use only.
    
    Reference: Florian Bloechl, Fabian J. Theis, Fernando Vega-Redondo,
    and Eric O'N. Fisher: Vertex Centralities in Input-Output
    Networks Reveal the Structure of Modern Economies, submitted. 
    '''

    n = A.shape[0] # number of nodes
    H = np.zeros((n, n)) # preallocate MFHT matrix
    M = np.linalg.inv(np.diag(np.sum(A, axis = 1))) @ A 
    A = np.eye(n) - M # compute transition matrix
    
    # I = np.linalg.inv(A[1:, 1:])
    # H = np.zeros((n, n)) # reallocate MFHT matrix
    
    e = np.ones((n-1,1))
    I = np.eye(n-1)

    for t in np.arange(n):
        SKIP_T = np.concatenate([np.arange(0, t), np.arange(t+1, n)], axis=None)
        M_t = M[SKIP_T][:, SKIP_T]
        H[:, t][SKIP_T] = ((np.linalg.inv(I - M_t)) @ e).reshape((n-1,))

    return H


def _minimum_edges(A, rnd_seed, corr_factor=np.array(1e-04, dtype=np.float64)):
    np.random.seed(rnd_seed)
    A = np.array(A, dtype = np.float64)
    values = np.random.uniform(corr_factor, corr_factor*np.array(2), A[np.where(A == 0)].shape)
    A[np.where(A == 0)] = values
    return A

def rwc(input_filepath, output_filepath, rnd_seed, weights=None):
    '''
    :param A: a Numpy array of dimension (N x N) describing an Input/Output Tables.

    :returns cen: a Numpy array of dimension N with the Random Walk Centrality of the sectors of A.

    Calculates random walk centrality cen for weighted directed 
    network with adjacency matrix a. The graph may contain self-loops.
    
    Porting to Python from the Matlab Script of
    Author: Florian Bloechl, July 2009 
    http://cmb.helmholtz-muenchen.de
    This software is for non-commercial use only.
    
    Reference: Florian Bloechl, Fabian J. Theis, Fernando Vega-Redondo,
    and Eric O'N. Fisher: Vertex Centralities in Input-Output
    Networks Reveal the Structure of Modern Economies, submitted.
    '''
  
    try:
        assert type(input_filepath) is np.ndarray
        A = input_filepath
    except AssertionError:
        A = np.load(input_filepath)
    # Checking if the matrix is square
    try:
        assert A.shape[0] == A.shape[1]
    except IndexError:
        n = np.int(np.sqrt(A.shape[0]))
        A = A.reshape(n, n)
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == n

    n = A.shape[0]
    A = _minimum_edges(A, rnd_seed)
    
    m = _mfpt(A)
    cen = np.power(np.average(m, weights=weights, axis=0), -1)
    np.save(output_filepath, cen)
    return cen