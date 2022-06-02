# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:57:25 2022

@author: Kristian Søgaard

Includes functions used to obtain the theoretical lower bound for ZDMD on AR(1) sources
and MSE distortion contraints.
"""

import numpy as np
from scipy.optimize import minimize


def sig2x_fun(a, sig2w):
    return sig2w / (1-a**2)


def lda_fun(a, pi, sig2w):
    return a**2 * pi + sig2w


def Central_D_Theta(a, pi, h, lda, rho, sig2w, sig2x):
    """
    Calculates the optimum theoretical central distortion, D0,
    and the theoretical central distortion if averaging only, D0_avg.
    Also determines the optimal central predictors, theta_0 and theta_alpha

    Covariances calculated according to Lemma 4.1
    Input:
        a         Source process AR parameter.
        b         Innovation scale parameter
        pi        Side distortion
        h         scaling coefficient
        rho...    Noise correlation
        sig2w.    Innovation process variance
        sig2x     Source process variance
    output:
        theta_0       Optimal central predictor
        theta_alpha   Optimal central error predictor
        D0            Optimal theoretical central distortion
        D0_avg        Theo. central distortion if average + scale
    """
    lda = lda_fun(a, pi, sig2w)

    Sig_XY = (h/(1-(a**2)*(1-h))) * sig2x

    Sig_X_VC = h*(sig2x - a**2*Sig_XY)

    Sig_Y = (h**2*sig2x + 2*(a**2)*h*(1-h)*Sig_XY + h*pi)/(1-a**2*(1-h)**2)

    Sig_Y_12 = (h**2*sig2x + 2*(a**2)*h*(1-h)*Sig_XY + rho*pi*h)/(1-a**2*(1-h)**2)

    # Sig_a_12 = (b**2*sig2w + a**2 * rho*pi*h) / (1 - a**2*(1-h)**2)
    Sig_a_12 = sig2x + a**2*Sig_Y_12 - a**2*2*Sig_XY
    Sig_a_Vc = 0.5*h*(lda + Sig_a_12)

    sig2a_tilde = h**2 * lda + pi*h

    Sig_Vc = (sig2a_tilde + h**2 * Sig_a_12 + rho*pi*h)/2

    theta_alpha = Sig_a_Vc / Sig_Vc

    Sig_X_Y_C = theta_alpha*Sig_X_VC + a**2*Sig_XY

    term2 = a**2*(Sig_Y + Sig_Y_12)/2
    term3 = h*theta_alpha*(a**2*Sig_XY - a**2*Sig_Y_12)
    Sig_Y_C = theta_alpha**2*Sig_Vc + term2 + term3

    theta_0 = Sig_X_Y_C / Sig_Y_C
    D0 = sig2x - Sig_X_Y_C**2 / Sig_Y_C
    Sig_Y_avg = (Sig_Y + Sig_Y_12)/2
    theta_0 = Sig_XY / Sig_Y_avg
    D0_avg = sig2x - Sig_XY**2 / Sig_Y_avg
    
    
    # if no packets received
    #d_lost = 0.5*a**2*pi + sig2w + 0.5*a**2*(sig2x + Sig_Y_12 - 2*Sig_XY)
    
    d_lost = sig2x- (a**2*Sig_XY**2*(2*Sig_Y - 2*Sig_Y_12))/(Sig_Y**2- Sig_Y_12**2)
    
    return theta_0, theta_alpha, D0, D0_avg, d_lost


### Lower bound ###
def scal_lb(pi, lda, rho):
    """
    The achievable rate in the test channel
    """
    return (np.log2(lda/pi) - 0.5*np.log2(1-rho**2))/2


### optimization variable splitting
def opt_var_split(x, pi=False):
    """
    Splits the optimization variable vector, x, into the two optimization variables rho0 and ds
    input:
        x    optimization variable vector
        pi   possible fixed side distortion
    output:
        rh0  correlation coefficient
        ds   side distortion
    """
    rho0 = x[0]
    if np.any(pi):
        ds = pi
    else:
        ds = x[1]
    return rho0, ds


### Objective function ####
def rho_objective(x, a, sig2w, pi=False):
    """
    The objective function. The objective is the scalar lower bound on the
    sum-rate, calculated in the scal_lb function.
    """
    rho0, ds = opt_var_split(x, pi)
    lda = lda_fun(a, ds, sig2w)
    return scal_lb(ds, lda, rho0)


###  Central distortion function###
def d0_func(x, a, sig2w, pi=False):
    """
    Determines the central distortion for the current optimization variable values.
    """
    rho0, ds = opt_var_split(x, pi)
    lda = lda_fun(a, ds, sig2w)
    h = 1 - ds / lda
    sig2x = sig2x_fun(a, sig2w)
    _, _, d0, _, _ = Central_D_Theta(a, ds, h, lda, rho0, sig2w, sig2x)

    return d0


### Optimization problem  ##
def optimization(D0, Ds, a, sig2w, pi=False, disp=True):
    """
    Solving the optimization problem (P_scal) for given source and distortion
    constraints.
    """
    ## Bounds and constraints ####
    bounds = [(-1, 0), (None, Ds)]
    con1 = {'type': 'ineq', 'fun': lambda d: D0 - d0_func(d, a, sig2w,
                                                          pi)}
    cons = [con1]

    # Initial guess
    x0 = np.array([-0.9, Ds/100])
    # print("Initial objective value:", rho_objective(x0, a, b, sig2w))
    # for i, c in enumerate(cons):
    #     print("Constraint: {:d}, value: {:.8f}".format(i, c['fun'](x0)))
    if pi:  # Fix side distortion to optimum
        bounds = [bounds[0]]
        x0 = x0[0]

    #### Solving optimization problem ####
    opt_res = minimize(rho_objective, x0, args=(a, sig2w, pi),
                       method='SLSQP', constraints=cons, bounds=bounds,
                       options={'disp': disp, 'maxiter': 3000})

    return opt_res


### Clean result ###
def clean_res(res, a, sig2w):
    rate = res['fun']
    opt_rho = res['x'][0]
    opt_ds = res['x'][1]
    opt_d0 = d0_func(res['x'], a, sig2w)
    flag = res['status']
    return (opt_d0, opt_ds, opt_rho, rate, flag)



#### Theoretical lower bound setup ####
def pi_s_func(Rs, rho, a, sig2w):
    """
    For given rate per description, Rs, and rho, determine corresponding pi_s.
    Since symmetric this is the same as the avg. sum rate.
    """
    pi = sig2w/(-a**2+np.exp(2*Rs*np.log(2)+(1/2)*np.log(-rho**2+1)))
    return pi



def rho_func(R, pi_s, a, sig2w):
    """
    For given rate per description, Rs, and pi_s, determine corresponding rho.
    Since symmetric this is the same as the avg. sum rate.
    """
    rho = -np.sqrt(1- 2**(-4*R)*((a**2*pi_s + sig2w)/pi_s)**2)
    return rho


def theo_dist_bound(Rs, a, sig2w, N=1000, Ds_bound=0, fix_rho=True):
    """
    For fixed rate per description, Rs, determine minimum D0 and Ds.
    """
    if fix_rho:
        rho_list = np.linspace(-1, 0, N)[1:]
        ds_list = pi_s_func(Rs, rho_list, a, sig2w)
        d0_list = d0_func([rho_list, ds_list], a, sig2w)
        d_lost = d_lost_fun([rho_list, ds_list], a, sig2w)

    else:
        assert(Ds_bound > 0)
        ds_list = np.linspace(0.0001, Ds_bound, N)
        rho_list = rho_func(Rs, ds_list, a, sig2w)
        d0_list = d0_func([rho_list, ds_list], a, sig2w)
        d_lost = d_lost_fun([rho_list, ds_list], a, sig2w)

    return d0_list, ds_list, d_lost



def lower_bound_packet_loss(Rs, a, sig2w, packet_loss_probs):
    d0_list, ds_list, d_lost = theo_dist_bound(Rs, a, sig2w, N=1000, Ds_bound=0, fix_rho=True)
    
    idx1 = d0_list > 0 
    idx2 = ds_list > 0
    idx = idx1*idx2
    d0_list, ds_list = d0_list[idx], ds_list[idx]
    
    
    lower_bound = np.zeros(len(packet_loss_probs))
    for i, prob in enumerate(packet_loss_probs):
        lower_bound_list = (1-prob)**2*d0_list + 2*(prob - prob**2)*ds_list + prob**2*d_lost
        lower_bound[i] = np.min(lower_bound_list)
        
    
    return lower_bound


def d_lost_fun(x, a, sig2w, pi=False):
    rho0, ds = opt_var_split(x, pi)
    lda = lda_fun(a, ds, sig2w)
    h = 1 - ds / lda
    sig2x = sig2x_fun(a, sig2w)
    _, _, _, _, d_lost = Central_D_Theta(a, ds, h, lda, rho0, sig2w, sig2x)
    return d_lost


def DRF(rates, a, sig2w):
    distortion = np.zeros(len(rates))
    for i, R in enumerate(rates):
        d0_list, ds_list, _ = theo_dist_bound(R, a, sig2w, N=1000, Ds_bound=0, fix_rho=True)
        distortion[i] = np.min(d0_list)
    
    return distortion



if __name__ == "__main__":
    # For a scalar valued AR(1) process X(k) = a*X(k-1) + *W(k)
    a = 0.9
    
    sig2w = 0.19
    sig2x = sig2x_fun(a, sig2w)
    Ds = 0.06445
    D0 = 0.003
    
    init_lda = lda_fun(a, Ds, sig2w)
    init_d0 = d0_func([0, Ds], a, sig2w)

    opt_res = optimization(D0, Ds, a, sig2w)
    opt_rho = opt_res['x'][0]
    try:
        opt_ds = opt_res['x'][1]
    except:
        opt_ds = Ds
    opt_lda = lda_fun(a, opt_ds, sig2w)
    opt_h = 1 - opt_ds / opt_lda
    opt_d0 = d0_func(opt_res['x'], a, sig2w)
    _, _, _, Rs, _ = clean_res(opt_res, a, sig2w)
    R0 = -0.5*np.log2(1-opt_rho**2)
    R1 = 0.5*np.log2(opt_lda/opt_ds)
    
    
    # Kristian 
    # sum_rate1 = np.zeros((100,100))
    # results = []
    
    # Ds_list = np.logspace(-4, 0, 100)
    # D0_super_list = []
    # for i, Ds in enumerate(Ds_list):
    #     D0_list = np.logspace(-6, np.log10(Ds), 100)
    #     D0_super_list.append(D0_list)
    #     for j, D0 in enumerate(D0_list):
    #         opt_res = optimization(D0, Ds, a, sig2w)
    #         results.append(opt_res)
    #         sum_rate1[i,j] = opt_res['fun']
    
    # Rs = 5
    # d0_list, ds_list, _ = theo_dist_bound(Rs, a, sig2w, N=1000, Ds_bound=0, fix_rho=True)
    
    # idx1 = d0_list > 0 
    # idx2 = ds_list > 0
    # idx = idx1*idx2
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.linspace(-1, 0, 999)[idx], 10*np.log10(d0_list[idx]))
    # plt.plot(np.linspace(-1, 0, 999)[idx], 10*np.log10(ds_list[idx]))
    # plt.show()
    
    # plt.figure()
    # plt.plot(10*np.log10(d0_list[idx]), 10*np.log10(ds_list[idx]))
    # plt.show()
    
    packet_loss_probs = np.linspace(0, 0.5, 50)
    # lower_bound = lower_bound_packet_loss(Rs, a, sig2w, packet_loss_probs)
    
    # plt.figure()
    # plt.plot(packet_loss_probs, 10*np.log10(lower_bound))
    # plt.show()
    
    
    
    # rates = np.linspace(2, 6, 20)
    # drf = DRF(rates, a, sig2w)
    # plt.figure()
    # plt.plot(10*np.log10(drf[2:]), rates[2:]*2)
    # plt.show()