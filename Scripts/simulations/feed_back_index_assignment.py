# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:59:05 2022

@author: Kristian Søgaard

Index assigment scheme for ZDMD coding. It is assumed that the encoder knows which descriptions
are received, and therefore effectively knows the reconstructions to any time.
"""

import numpy as np



def index_assignment(b, index_map, Delta_C):
    shift = index_map.shape[0]*Delta_C
    
    n_shifts = np.round(b/shift)
    
    index_map = shift*n_shifts + index_map*Delta_C
    
    a = index_map[np.isclose(index_map[:,0], b, rtol=1e-9, atol=0.0), 1:]
    
    return a
    
    
def encoder(X, Y_old, a, Delta_C, index_map):
    """
    Encoder function for index assignment scheme.

    Parameters
    ----------
    X : float
        Source sample.
    Y_old : ndarray
        past p samples.
    a : ndarray
        The p AR(p) coefficients.
    Delta_C : float
        Bin size for central decoder.
    index_map : ndarray
        Matrix with index assigment table. First column is central quantization
        and the rest are the associated assigned indexes.

    Returns
    -------
    q : ndarray
        Side descriptions.
    shift : float
        How much to add for one shift.
    n_shifts : int
        number of shifts needed.
    U : float
        prediction error sample.

    """
    
    U = X - a.T @ Y_old
    
    # Central quantizer 
    b = np.round(U/Delta_C)*Delta_C 
    
    # Index assigment
    q = index_assignment(b, index_map, Delta_C)
    
    Y = b + a.T @ Y_old
    
    return q, U, b, Y


def encoder_2(X, Y_old, a, Delta_C, index_map):
    """
    Encoder function for index assignment scheme.

    Parameters
    ----------
    X : float
        Source sample.
    Y_old : ndarray
        past p samples.
    a : ndarray
        The p AR(p) coefficients.
    Delta_C : float
        Bin size for central decoder.
    index_map : ndarray
        Matrix with index assigment table. First column is central quantization
        and the rest are the associated assigned indexes.

    Returns
    -------
    q : ndarray
        Side descriptions.
    shift : float
        How much to add for one shift.
    n_shifts : int
        number of shifts needed.
    U : float
        prediction error sample.

    """
    
    U = X - a.T @ Y_old
    
    # Central quantizer 
    b = np.round(U/Delta_C)*Delta_C 
    
    # Index assigment
    q = index_assignment(b, index_map, Delta_C)
    
    Y = np.mean(q) + a.T @ Y_old
    
    return q, U, b, Y



def decoder(q, index_map, Delta_C, a, Y_old, received_packet):
    """
    

    Parameters
    ----------
    q : ndarray
        Side descriptions.
    index_map : ndarray
        Matrix with index assigment table. First column is central quantization
        and the rest are the associated assigned indexes.
    Delta_C : float
        Bin size for central decoder.
    Y_old : ndarray
        past p samples.
    a : ndarray
        The p AR(p) coefficients.
    received_packet : ndarray, bool
        Boolean array. If an entry is True, then description is received.

    Returns
    -------
    Y : float
        Source sample reconstruction.

    """
    shift = index_map.shape[0]*Delta_C
    
    
    if received_packet.all():
        n_shifts = 0
        index = shift*n_shifts + index_map*Delta_C
        b = index[(index[:,1:] == q).all(axis=1), 0]
        
        while b.shape != (1,):
            sgn = np.sign(q[0])
            n_shifts += sgn*1
            index = shift*n_shifts + index_map*Delta_C
            b = index[(index[:,1:] == q).all(axis=1), 0]
            
            
            
        
    elif received_packet.any():
        b = q[received_packet]
    else:
        b = 0
    
    
    # print(received_packet)
    
    # print(Delta_C)
    # print(q)
    # print(b)
    Y_side = q + a.T @ Y_old
    Y = b + a.T @ Y_old
    return Y, Y_side


def run_index_assigment(X, Delta_C, index_map, a, packet_loss_prob, return_b = False):
    p = a.shape[0]
    
    N = len(X)
    q = np.zeros((N,2))
    U = np.zeros(N)
    b = np.zeros(N)
    Y = np.zeros(N+p)
    Y_side = np.zeros((N,2))
    
    a = np.flip(a)
    a = a.reshape((p, 1))
    
    Y_old = np.zeros(p)
    
    for i in range(N):
        q[i,:], U[i], b[i], Y_enc = encoder(X[i], Y_old, a, Delta_C, index_map)
        
        if packet_loss_prob != 0:
            received_packet = np.random.uniform(size=2) > packet_loss_prob
        else:
            received_packet = np.ones(2, np.bool)
        
        Y[i+p], Y_side[i,:] = decoder(q[i,:], index_map, Delta_C, a, Y_old, received_packet)
                
        Y_old = Y[i+1:i+p+1]
        
    Y = Y[p:]
    if return_b:
        return q, Y, U, Y_side, b
    else:
        return q, Y, U, Y_side




def binsize_index_assignment(R, a, sig2w, r=3):
    nom = 12*2*np.pi*np.e*sig2w
    denom = 12*2**R*r**2 - 2*np.pi*np.e*np.linalg.norm(a)
    
    Delta_C = np.sqrt(nom/denom)
    
    
    return Delta_C
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import AR_process as AR
    import entropy_coder as ec
    
    # # Parameters #
    init_seed = 123456
    np.random.seed(init_seed)
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    N_time = 100000
    
    r = AR.get_acf_from_params(a, sig2w)
    X = AR.generate_AR_process(r, N_time)[0]
    
    a, sig2w = AR.find_params(r[:2])
    packet_loss_prob = 0
    sum_rate = 12
    nesting_ratio = 7
    Delta_C = binsize_index_assignment(sum_rate, a, sig2w, r=nesting_ratio)
    
    index = np.genfromtxt(f"tables/IA_matrix_{nesting_ratio}_2.csv", delimiter=',')
    

    q, Y, U, Y_side, b  = run_index_assigment(X, Delta_C, index, a, packet_loss_prob, return_b = True)
    
    var_U = np.var(U)
    
    rate = np.log2(2*np.pi*np.e*var_U)-2*np.log2(3*Delta_C)
    
    Z1 = U[2:] - q[2:,0]
    Z2 = U[2:] - q[2:,1]
    rho_est = np.corrcoef((Z1,Z2))[1,0]
    print(f"Quantization noise correlation = {rho_est}")
    
    plt.figure()
    plt.plot(X)
    plt.plot(Y_side)
    plt.plot(Y)
    plt.show()
    
    plt.figure()
    plt.plot((X - Y)**2)
    plt.show()
    
    ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(q[2:, 0], decimals=12), len(q[2:, 0]))
    ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(q[2:, 1], decimals=12), len(q[2:, 1]))
    
    bitrate = bitrate_1 + bitrate_2
    
    distortion = 10*np.log10(np.mean((X-Y)**2))
    
    
    # Histogram
    bins, counts = np.unique(b, axis=0, return_counts=True)
    no_bins = len(counts)
    pn = counts/len(b)
    
    plt.figure()
    plt.stem(bins, pn)
    plt.show()
    
    
    import statsmodels.api as sm
    import pylab as py
    
    
    sm.qqplot(U, line ='45')
    py.show()
    
    br = AR.autocorr(b)
    plt.figure()
    plt.stem(br)
    plt.show()
    
    
    
    
    
    
