"""
Created on Wed May 13 14:59:05 2022

@author: Kristian Søgaard

Staggered Scheme using optimal predictors. It is assumed that the encoder knows which descriptions
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


def stag_encode(X, a, Y_old, Delta_S, Delta_0):
    """
    Encoding of source sample X using the staggered scheme. 

    Parameters
    ----------
    X : float
        Source sample.
    a : array
        Source coefficents.
    Y_old : array 
        Side reconstruction samples to p previous time steps.
    Delta_S: Quantizer stepsize for first stage quantizer
    Delta_0: Quantizer stepsize for refinement quantizer

    Returns
    -------
    U_quant : array
        Output of side quantizers.
    e_c_quant : float
        Output of central quantizer.
    Y_new : array
        Side reconstructions.

    """
    
    # Side quantization
    sgn = np.sign(Y_old[:,0] - Y_old[:,1])
    shift = Delta_S/2*(1 - a.T @ sgn)[0]
    U = (X - a.T @ Y_old)[0]
    U_quant = quantization(U, Delta_S, shift)
    

    # Reconstruction
    Y_new = (U_quant + a.T @ Y_old)[0]
    Y_0_C = np.mean(Y_new)
    
    # Central quantization
    e_c = X - Y_0_C
    e_c_quant = np.round((e_c)/Delta_0)*Delta_0
    
    return U_quant, e_c_quant, Y_new


def stag_decode(U_quant, e_c_quant, a, Y_old, Delta_S, Delta_0, received_packet):
    """
    Decoding functions of staggered scheme. 

    Parameters
    ----------
    U_quant : array
        Output of side quantizers.
    e_c_quant : float
        Output of central quantizer.
    a : array
        Source coefficents.
    Y_old : array 
        Side reconstruction samples to p previous time steps.
    Delta_S: Quantizer stepsize for first stage quantizer
    Delta_0: Quantizer stepsize for refinement quantizer
    received_packet : boolean array
        Array deciding which packets are recieved. True = Recieved.

    Returns
    -------
    Y_0 : float
        Central reconstruction.
    Y_new : array
        Side reconstructions.
    Y_used : float
        The used reconstruction. Depends on received_packet.
    """
    
    
    Y_new = (U_quant + a.T @ Y_old)[0]
    
        
    if not received_packet.all(): 
        if received_packet.any():  # if one packet received          
            Y_0 = 0
            Y_used = Y_new[received_packet]
            
        elif not received_packet.any(): # if no packets received
            Y_0 = 0
            Y_used = np.mean(Y_new)
    
            
    if received_packet.all(): # If both packets received. 
        Y_0_C = np.mean(Y_new)
        Y_0 = Y_0_C + (e_c_quant)
        Y_used = Y_0
    
    return Y_0, Y_new, Y_used



def fix_stag_setup(R1, R0, a, sig2w): 
    """
    Setup of the staggered scheme. Works for ar(1) and ar(p)
    
    Inputs:
        R1:   Rate of first stage quantizer (float)
        R0:   Rate of refimement quantizer (float)
        a:    Source coefficients (array)
        sig2w White Gaussian noise variance for source (float)
    
    Returns:
        Delta_S: Quantizer stepsize for first stage quantizer
        Delta_0: Quantizer stepsize for refinement quantizer
        pi_s:    Theoretic (approximate) side MSE distortion
        pi_0:    Theoretic (approximate) central MSE distortion
    """
    
    k = np.e*np.pi*2
    
    nom = 2*np.sqrt(3)*np.sqrt((12*4**R1-k*np.linalg.norm(a)**2)*k*sig2w) 
    denom = 12*4**R1-k*np.linalg.norm(a)**2
    Delta_S = nom/denom
        

    pi_s = Delta_S**2/12

    pi_0 = pi_s / 4
    Delta_0 = 2**(-R0)*np.sqrt(12*pi_0)

    return Delta_S, Delta_0, pi_s, pi_0


def run_stag_scheme(X, R1, R0, a, sig2w, packet_loss_prob):
    """
    For a source sequence X, and a rate pair (R1, R0) compute the recontruction
    sequences. 
    
    IT IS ASSUMED THAT THE ENCODER AND DECODER ARE SYNCHRONIZED. THUS THEY USE 
    THE SAME Y_OLD EVEN IF PACKET LOSSES OCCUR. 

    Parameters
    ----------
    X : array
        Soruce sequence.
    R1:   Rate of first stage quantizer (float)
    R0:   Rate of refimement quantizer (float)
    a : array
        Source coefficents.
    sig2w : float
        White Gaussian noise variance for source.
    packet_loss_prob : float 
        packet loss probability 0 <= packet_loss_prob < 1.

    Returns
    -------
    U_quant : array
        Output of side quantizers.
    e_c_quant : float
        Output of central quantizer.
    Y_0 : array
        Central reconstruction sequence.
    Y_side : array
        Side reconstruction sequences.
    Y_used : array
        Used reconstruction sequence.

    """
    
    p = len(a)
    N = len(X)
    U_quant = np.zeros((N,2))
    e_c_quant = np.zeros(N)
    Y_0 = np.zeros(N)
    Y_used = np.zeros(N)
    Y_side = np.zeros((p+N,2))
    U_received = np.zeros((N,2))
    
    a = np.flip(a)
    a = a.reshape((p, 1))
    
    Delta_S, Delta_0 = fix_stag_setup(R1, R0, a, sig2w)[:2]
    Y_old = np.zeros((p, 2))
    
    
    
    for i in range(N):

        U_quant[i,:], e_c_quant[i], _ = stag_encode(X[i], a, Y_old, Delta_S, Delta_0)
        
        
        received_packet = np.random.uniform(size=2) > packet_loss_prob
        
        if received_packet.any():
        
            U_received[i,:] = U_quant[i,:]*received_packet
            Y_0[i], Y_side[p+i,:], Y_used[i] = stag_decode(U_received[i,:], e_c_quant[i], a, Y_old,
                                                         Delta_S, Delta_0, received_packet)
        else: 
            Y_0[i], Y_side[p+i,:], Y_used[i] = Y_0[i-1], Y_side[i-1,:], Y_used[i-1]
        
        Y_old = Y_side[i+1:i+p+1,:]
        
    Y_side = Y_side[p:,:]
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
    N_time = 10000
    
    r = AR.get_acf_from_params(a, sig2w)
    X = AR.generate_AR_process(r, N_time)[0]
    
    #a, sig2w = AR.find_params(r[:2])
    
    R0 = 3.5
    R1 = 3
    packet_loss_prob = 0
    
    #U_quant, e_c_quant, Y_0, Y_side, Y_used, Y_old_enc_list, Y_old_dec_list = run_samacheme_no_feedback(X, R1, R0, a, sig2w, packet_loss_prob)
    
    U_quant, e_c_quant, Y_0, Y_side, Y_used = run_stag_scheme(X, R1, R0, a, sig2w, packet_loss_prob)
    ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(U_quant[2:, 0], decimals=12), len(U_quant[2:, 0]))
    ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(U_quant[2:, 1], decimals=12), len(U_quant[2:, 1]))
    ent_3, avg_rate_3, bitrate_3 = ec.entropy_coder(np.round(e_c_quant[2:], decimals=12), len(e_c_quant[2:]))
        
    
    # plt.figure()
    # plt.plot(X)
    # plt.plot(Y_side)
    # plt.plot(Y_0)
    # plt.grid()
    # plt.show()
    
    # plt.figure()
    # plt.plot(X)
    # plt.plot(Y_used)
    # plt.show()
    
    # plt.figure()
    # plt.plot((X - Y_used)**2)
    # plt.show()
    
    R1 = 0.8
    Delta_S, Delta_0,_,_ = fix_stag_setup(R1, R0, a, sig2w)
    
    R1_after = 0.5*np.log2(2*np.pi*np.e*(Delta_S**2/12*np.linalg.norm(a)**2 + sig2w)) - np.log2(Delta_S)
    
