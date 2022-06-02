# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:08:33 2021

@author: Kristian Søgaard

Contains functions for generating AR(p) processes given samples of ACS 
and function for estimating the model coeficients.
"""

import numpy as np
import scipy.linalg as sl


def gaussian_process(samples, a, var_z, r):
    """ 
    Generate AR(p) process with 'samples' samples, where
    p = len(r)-1 given p+1 samples of ACF, AR coefficients, and noise variance. 
    r[0] is E[x[0]x[0]].
    
    Parameters
    ----------
        samples :   number of samples in returned sequence
        a :         model coefficients
        var_z :     noise variance
        r :         p+1 first samples of ACF
        
    Returns
    -------
        x :         samples of AR(p) process with p = len(a)
    
    """
    
    p = len(a)
    
    var_x = get_process_var(a, var_z, r)
    
    x_ini = np.random.randn(p)*np.sqrt(var_x)
    
    x = np.zeros(samples - p)
    x = np.concatenate((x_ini, x))
        
    for t in range(p, samples):
        for k in range(p):
            x[t] = x[t] + a[k]*x[t-(k+1)]
        x[t] = x[t] + np.random.randn(1)*np.sqrt(var_z)
    
    return x


def get_process_var(a, var_z, r):
    """
    Calculate the variance of the stationary AR(p) process with parameters 
    a, var_z and ACF r.
    
    Parameters
    ----------
        a :         model coefficients
        var_z :     noise variance
        r :         p+1 first samples of ACF
        
    Returns
    -------
        var_x :     variance of stationary AR(p) process
    """
    
    var_x = var_z
    for i in range(len(a)):
        var_x += a[i]*r[i+1]
        
    return var_x


def find_params(r):
    """
    Uses Yule-Walker equations to find parameters for AR(p) process, 
    where p = len(r) - 1 
    
    Parameters
    ----------
        r :         p+1 first samples of ACF
        
    Returns
    -------
        a :         model coefficients, (p,)
        var_z :     noise variance
    """
    
    a = sl.solve_toeplitz(r[:-1], r[1:])
    
    var_z = r[0]
    for i in range(len(a)):
        var_z += -a[i]*r[i+1]
        
    return a, var_z


def levinson_durbin(r):
    """
    Uses Levinson Durbin algorithm to solve for AR coefficients

    Parameters
    ----------
        r :     p+1 first samples of autocorrelation function.

    Returns
    -------
        a :     model coefficients, (p,)

    """
    a = sl.solve_toeplitz(r[:-1], r[1:])
    return a


def autocorr(x, onesided=False):
    """
    Biased estimate the ACF for stationary process with samples x.
    
    Parameters
    ----------
        x :         samples of process for which the ACF should be estimated
        onesided:   whether to only return ACF for positive lags. Default False
        
    Returns
    -------
        result :    Estimate of ACF
        
    """
    samples = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')/len(x)
    if onesided:
        return result[samples-1:]
    return result


def generate_AR_process(r, samples):
    """ 
    Given samples of ACF, generate AR(p) process with 'samples' samples, where
    p = len(r)-1. r[0] is E[x[0]x[0]]
    
    Parameters
    ----------
        r :         p+1 first samples of ACF
        samples :   number of samples in returned sequence
        
    Returns
    -------
        x :         samples of AR(p) process with p = len(a)
    
    """
    
    a, var_z = find_params(r)
    x = gaussian_process(samples, a, var_z, r)
    
    return x, a, var_z


def get_acf_from_params(a, sigw):
    """
    Calculates the first p+1 values of the ACF for the AR(p) process with
    coefficients a and additive noise variance sigw

    Parameters
    ----------
    a : ndarray
        coefficients of process.
    sigw : float
        variance of noise process.

    Returns
    -------
    r : ndarray
        first p+1 values of ACF.

    """
    
    b = np.zeros(len(a)+1)
    b[0] = -sigw
    
    C = np.zeros((len(a)+1, len(a)+1))
    C[0,:] = np.concatenate((np.array([-1]), a))
    C[1:,0] = a
    
    zero_blocks = np.zeros((len(a)+1, len(a)+1))
    C = np.block([zero_blocks, C, zero_blocks])
    
    
    for i in range(1,len(a)+1):            
        for j in range(len(a)+2,2*(len(a)+1)):
            C[i,j] = C[0,i+j] + C[0,2*(len(a)+1)-(j-i)]
            
    C = C[:,len(a)+1:2*(len(a)+1)]
    
    r = np.linalg.solve(C,b)
    
    return r


def main():
    """
    Examples of usage of module
    """
    import matplotlib.pyplot as plt
    np.random.seed(1)
    r = np.array([1, 0.5, 0.3])
    
    
    ### Verify correct simulation of AR process
    samples = 1000
    realizations = 1000
    
    acs_rel = np.zeros((realizations,2*samples-1))
    for i in range(realizations):
        gaus, _, _ = generate_AR_process(r, samples)
        acs_rel[i,:] = autocorr(gaus)
    
    acs_mean = np.mean(acs_rel, 0)
    acs_var = np.var(acs_rel,0)
    
    
    plt.figure()
    plt.plot(acs_mean, label="ACF Estimate")
    plt.xlabel("k")
    plt.ylabel("r[k]")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(acs_mean[samples-1:samples+20], label="ACF Estimate")
    plt.plot(np.arange(len(r)), r, "ko", label="True ACF")
    plt.xlabel("k")
    plt.ylabel("r[k]")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(gaus)
    plt.xlabel("t")
    plt.ylabel("x[k]")
    plt.title(f"Realization of AR({len(r)-1}) process")
    plt.show()
    
    
    ### Estimate coefficients using Levinson-Durbin algorithm (LPC) given true ACF
    a_2 = levinson_durbin(r) # True model order
    a_1 = levinson_durbin(r[:2]) # Lower model order
    print("LPC coefficients for different model orders")
    print(f"p = 2:   {a_2}")
    print(f"p = 1:   {a_1}")
    print(" ")
    
    ### Estimate coefficients using Levinson-Durbin algorithm (LPC) given samples x
    samples = 100
    x, _, _ = generate_AR_process(r, samples)
    r_est = autocorr(x, onesided=True)
    a_2 = levinson_durbin(r_est[:3]) # True model order
    a_1 = levinson_durbin(r_est[:2]) # Lower model order
    print("LPC coefficients for different model orders")
    print(f"p = 2:   {a_2}")
    print(f"p = 1:   {a_1}")
    
    
if __name__ == "__main__":
    #main()
    
    r = np.array([1, 0.9, 0.7])
    gaus, a, var = generate_AR_process(r, 1000)
    
    



