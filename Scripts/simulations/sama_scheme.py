# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:59:05 2022

@author: Kristian Søgaard

Scheme using staggering and sub-optimal predictors for ZDMD coding. 
It is assumed that the encoder knows which descriptions
are received, and therefore effectively knows the reconstructions to any time.
"""



import numpy as np


def quantization(signal, Delta, q):
    """
    Staggered quantization of two inputs
    Input:
        signal   Array of shape (2,)
        Delta    Quantization interval
        q        Quantization interval overlap
    output:
        Q_out    Quantized signal, array of shape (2,)
    """
    Q_out = np.empty(signal.shape)
    Q_out[0] = np.round(signal[0]/Delta)*Delta
    Q_out[1] = np.round((signal[1] + q)/Delta)*Delta - q
    return Q_out


def sama_encode(X, Y, Delta_S, Delta_0, stag=True):

    if stag:
        shift = Delta_S/2
    else:
        shift = 0
    
    U = X - Y
    
    
    U_quant = quantization(U, Delta_S, shift)
    

    # Reconstruction
    Y_new = U_quant + Y

    Y_0_C = np.mean(Y_new)
    e_c = X - Y_0_C
    


    e_c_quant = np.round((e_c)/Delta_0)*Delta_0
    
    
    return U_quant, e_c_quant, Y_new, U


def sama_decode(U_quant, e_c_quant, Y, Delta_S, Delta_0, received_packet, lost_history=None):
    
    Y_new = np.zeros(2)
    
    
        
    Y_new = U_quant + Y
        
    if not received_packet.all():
        if received_packet.any():            
            Y_0 = 0
            Y_used = Y_new[received_packet]
            
        elif not received_packet.any():
            Y_0 = 0
            Y_used = np.mean(Y_new)
    
            
    if received_packet.all():
        Y_0_C = np.mean(Y_new)
        Y_0 = Y_0_C + (e_c_quant)
        Y_used = Y_0
    
    return Y_0, Y_new, Y_used



def fix_wickrama_setup(R1, R0, a, sig2w, optimal=False): 
    """
    Setup of the Samarawickrama scheme. Works for ar(1) and ar(p)
    """
    k = np.e*np.pi*2
    a = a[0]

    if optimal:
        lda = sig2w
        nom = 2*np.sqrt(3)*np.sqrt(-(k*a**2-12*4**R1)*k*sig2w) 
        denom = k*a**2 - 12*4**R1
        print("high rate Delta_S", 2**(np.log2(k*lda) - R1))
        Delta_S = -nom/denom


    else:
        lda = 2/(1+a) * sig2w

        ### High Rate assumption ####
        # Solving R=(1/2)*log(k*lda) - log(Delta):
        # Delta_S = 2**(-R1)*np.sqrt(k*lda)

        ### Additive uniform noise assumption  ###
        # Solving R=(1/2)*log(k*(lda+Delta^2/12))-log(Delta):
        Delta_S = np.sqrt(12*k*lda/(12*2**(2*R1)-k))
    
    pi_s = Delta_S**2/12

    pi_0 = pi_s / 4
    Delta_0 = 2**(-R0)*np.sqrt(12*pi_0)

    return Delta_S, Delta_0, pi_s, pi_0


def run_samacheme(X, R1, R0, a, sig2w, packet_loss_prob, print_cor=False):
    
    N = len(X)
    U_quant = np.zeros((N,2))
    U = np.zeros((N,2))
    e_c_quant = np.zeros(N)
    Y_0 = np.zeros(N)
    Y_used = np.zeros(N)
    Y_side = np.zeros((N,2))
    U_received = np.zeros((N,2))
    
    Delta_S, Delta_0 = fix_wickrama_setup(R1, R0, a, sig2w)[:2]
    Y_old = np.zeros(2)
    
    
    
    for i in range(N):
        if i == 0:
            U_quant[i,:], e_c_quant[i], _, U[i,:] = sama_encode(X[i], Y_old, Delta_S, Delta_0, 
                                                     stag=True)
        else:
            U_quant[i,:], e_c_quant[i], _, U[i,:] = sama_encode(X[i], Y_old, Delta_S, Delta_0, 
                                                     stag=False)
        
        
        received_packet = np.random.uniform(size=2) > packet_loss_prob
        
        if received_packet.any():
        
            U_received[i,:] = U_quant[i,:]*received_packet
            Y_0[i], Y_side[i,:], Y_used[i] = sama_decode(U_received[i,:], e_c_quant[i], Y_old,
                                                         Delta_S, Delta_0, received_packet)
        else: 
            Y_0[i], Y_side[i,:], Y_used[i] = Y_0[i-1], Y_side[i-1,:], Y_used[i-1]
        
        Y_old = Y_side[i,:]
    
    if print_cor:
        Z1 = U[2:,0] - U_quant[2:,0]
        Z2 = U[2:,1] - U_quant[2:,1]
        rho_est = np.corrcoef((Z1,Z2))[1,0]
        print(f"Quantization noise correlation = {rho_est}")
    
    return U_quant, e_c_quant, Y_0, Y_side, Y_used





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import AR_process as AR
    import entropy_coder as ec
    
    # Parameters #
    init_seed = 123456
    np.random.seed(init_seed)
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    N_time = 100000
    
    r = AR.get_acf_from_params(a, sig2w)
    X = AR.generate_AR_process(r, N_time)[0]
    
    a, sig2w = AR.find_params(r[:2])
    
    R0 = 3.5
    R1 = 3
    packet_loss_prob = 0
    
    
    U_quant, e_c_quant, Y_0, Y_side, Y_used = run_samacheme(X, R1, R0, a, sig2w, packet_loss_prob, print_cor=True)
    ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(U_quant[2:, 0], decimals=12), len(U_quant[2:, 0]))
    ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(U_quant[2:, 1], decimals=12), len(U_quant[2:, 1]))
    ent_3, avg_rate_3, bitrate_3 = ec.entropy_coder(np.round(e_c_quant[2:], decimals=12), len(e_c_quant[2:]))
    
    
    
    
    plt.figure()
    plt.plot(X)
    plt.plot(Y_side)
    plt.plot(Y_0)
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(X)
    plt.plot(Y_used)
    plt.show()
    
    plt.figure()
    plt.plot((X - Y_used)**2)
    plt.show()
    
    
    
