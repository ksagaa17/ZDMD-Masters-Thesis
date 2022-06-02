# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:59:05 2022

@author: Kristian Søgaard

Compare the theoretical distortions
"""

import numpy as np


def fix_stag_setup(R1, R0, a, sig2w): 
    """
    Setup of the Samarawickrama scheme. Works for ar(1) and ar(p)
    """
    
    k = np.e*np.pi*2
    
    nom = 2*np.sqrt(3)*np.sqrt((12*4**R1-k*np.linalg.norm(a)**2)*k*sig2w) 
    denom = 12*4**R1-k*np.linalg.norm(a)**2
    Delta_S = nom/denom
        

    pi_s = Delta_S**2/12

    pi_0 = pi_s / 4
    Delta_0 = 2**(-R0)*np.sqrt(12*pi_0)

    return Delta_S, Delta_0, pi_s, pi_0


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


def distortions(R, a, sig2w, N, shift_rate=False):
    if shift_rate:
        R = R-1
    R0_list = np.linspace(0, R-4, N)
    R1_list = (R - R0_list)/2
    
    side_dist = np.zeros(len(R0_list))
    cent_dist = np.zeros(len(R0_list))
    
    for i, R0 in enumerate(R0_list):
        R1 = R1_list[i]
        
        Delta_S, Delta_0, pi_s, pi_0 = fix_stag_setup(R1, R0, a, sig2w)
        side_dist[i] = pi_s
        cent_dist[i] = 2**(-2*R0)*pi_0
    
    return side_dist, cent_dist


def distortions_sama(R, a, sig2w, N):
    R0_list = np.linspace(0, R-4, N)
    R1_list = (R - R0_list)/2

    
    side_dist = np.zeros(len(R0_list))
    cent_dist = np.zeros(len(R0_list))
    
    for i, R0 in enumerate(R0_list):
        R1 = R1_list[i]
        
        Delta_S, Delta_0, pi_s, pi_0 = fix_wickrama_setup(R1, R0, a, sig2w)
        side_dist[i] = pi_s
        cent_dist[i] = 2**(-2*R0)*pi_0
    
    return side_dist, cent_dist


def binsize_index_assignment(R, a, sig2w, r=3):
    nom = 12*2*np.pi*np.e*sig2w
    denom = 12*2**R*r**2 - 2*np.pi*np.e*np.linalg.norm(a)
    
    Delta_C = np.sqrt(nom/denom)
    return Delta_C

def IA_distortions(R, a, sig2w, N):
    nesting_ratios  = [2*(i+1)+1 for i in range(N)]
    
    side_dist = np.zeros(len(nesting_ratios))
    cent_dist = np.zeros(len(nesting_ratios))
    
    for i, r in enumerate(nesting_ratios):
        Delta_C = binsize_index_assignment(R, a, sig2w, r=r)
        Delta_S = r*Delta_C
        
        cent_dist[i] = Delta_C**2/12
        side_dist[i] = Delta_C**2/12 + Delta_S**4/(48*Delta_C**2)
    
    
    return side_dist, cent_dist


def lower_bound_distortions(R, a, sig2w):
    d0_list_theo_1, ds_list_theo_1, _ = lb.theo_dist_bound(R/2, a, sig2w, N=1000, Ds_bound=1, fix_rho=True)
    d0_list_theo_2, ds_list_theo_2, _ = lb.theo_dist_bound(R/2, a, sig2w, N=1000, Ds_bound=0.01, fix_rho=False)
    d0_list_theo = np.concatenate((np.flip(d0_list_theo_2[ds_list_theo_2>ds_list_theo_1.max()]), d0_list_theo_1))
    ds_list_theo = np.concatenate((np.flip(ds_list_theo_2[ds_list_theo_2>ds_list_theo_1.max()]), ds_list_theo_1))
    
    
    
    return ds_list_theo, d0_list_theo

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import AR_process as AR
    import lower_bound as lb
    import seaborn as sns
    
    
    # =============================================================================
    #  central vs side distortion where no rate is used on the shift   
    # =============================================================================
    rates = [8, 10, 12]
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    r = AR.get_acf_from_params(a, sig2w)
    
    N = 10
    
    side_dist_stag2 = np.zeros((len(rates),N))
    cent_dist_stag2 = np.zeros((len(rates),N))
    
    for i, R in enumerate(rates):
        side_dist_stag2[i,:], cent_dist_stag2[i,:] = distortions(R, a, sig2w, N)
    
    
    a, sig2w = AR.find_params(r[:2])
    side_dist_stag1 = np.zeros((len(rates),N))
    cent_dist_stag1 = np.zeros((len(rates),N))
    
    side_dist_sama = np.zeros((len(rates),N))
    cent_dist_sama = np.zeros((len(rates),N))
    
    for i, R in enumerate(rates):
        side_dist_stag1[i,:], cent_dist_stag1[i,:] = distortions(R, a, sig2w, N)
        side_dist_sama[i,:], cent_dist_sama[i,:] = distortions_sama(R, a, sig2w, N)
        
    colors = ['r', 'b', 'g', 'k']
    plt.figure()
    for i, R in enumerate(rates):
        
        plt.plot(10*np.log10(cent_dist_stag2[i,:]), 10*np.log10(side_dist_stag2[i,:]), color=colors[i], linestyle="dashed")
        plt.plot(10*np.log10(cent_dist_stag1[i,:]), 10*np.log10(side_dist_stag1[i,:]),  color=colors[i], linestyle ="dotted")
        plt.plot(10*np.log10(cent_dist_sama[i,:]), 10*np.log10(side_dist_sama[i,:]),  color=colors[i])
    
    plt.xlabel(r"$D_0$")
    plt.ylabel(r"$D_S$")
    plt.show()
    
    
    side_dist_theo = np.zeros((len(rates),N))
    cent_dist_theo = np.zeros((len(rates),N))
    # =============================================================================
    #   central vs side distortion where  rate is used on the shift   
    # =============================================================================
    rates = [8, 10, 12]
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    r = AR.get_acf_from_params(a, sig2w)
    
    N = 10
    
    side_dist_stag2 = np.zeros((len(rates),N))
    cent_dist_stag2 = np.zeros((len(rates),N))
    
    for i, R in enumerate(rates):
        side_dist_stag2[i,:], cent_dist_stag2[i,:] = distortions(R, a, sig2w, N,shift_rate=True)
    
    
    a, sig2w = AR.find_params(r[:2])
    side_dist_stag1 = np.zeros((len(rates),N))
    cent_dist_stag1 = np.zeros((len(rates),N))
    
    side_dist_sama = np.zeros((len(rates),N))
    cent_dist_sama = np.zeros((len(rates),N))
    
    for i, R in enumerate(rates):
        side_dist_stag1[i,:], cent_dist_stag1[i,:] = distortions(R, a, sig2w, N, shift_rate=True)
        side_dist_sama[i,:], cent_dist_sama[i,:] = distortions_sama(R, a, sig2w, N)
        
    colors = ['r', 'b', 'g', 'k']
    plt.figure()
    for i, R in enumerate(rates):
        
        plt.plot(10*np.log10(cent_dist_stag2[i,:]), 10*np.log10(side_dist_stag2[i,:]), color=colors[i], linestyle="dashed")
        plt.plot(10*np.log10(cent_dist_stag1[i,:]), 10*np.log10(side_dist_stag1[i,:]),  color=colors[i], linestyle ="dotted")
        plt.plot(10*np.log10(cent_dist_sama[i,:]), 10*np.log10(side_dist_sama[i,:]),  color=colors[i])
    
    plt.xlabel(r"$D_0$")
    plt.ylabel(r"$D_S$")
    plt.show()
    
    
    # =============================================================================
    #  IA   
    # =============================================================================
    rates = [8, 10, 12]
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    r = AR.get_acf_from_params(a, sig2w)
    
    N = 7
    
    side_dist_IA2 = np.zeros((len(rates),N))
    cent_dist_IA2 = np.zeros((len(rates),N))
    
    for i, R in enumerate(rates):
        side_dist_IA2[i,:], cent_dist_IA2[i,:] = IA_distortions(R, a, sig2w, N)
    
    
    a, sig2w = AR.find_params(r[:2])
    side_dist_IA1 = np.zeros((len(rates),N))
    cent_dist_IA1 = np.zeros((len(rates),N))
    
    
    for i, R in enumerate(rates):
        side_dist_IA1[i,:], cent_dist_IA1[i,:] = IA_distortions(R, a, sig2w, N)
        
    # colors = ['r', 'b', 'g', 'k']
    
    # plt.figure()
    # for i, R in enumerate(rates):
    #     ds_list_theo, d0_list_theo = lower_bound_distortions(R, a, sig2w)
    #     plt.plot(10*np.log10(cent_dist_IA2[i,:]), 10*np.log10(side_dist_IA2[i,:]), color=colors[i], linestyle="dashed")
    #     plt.plot(10*np.log10(cent_dist_IA1[i,:]), 10*np.log10(side_dist_IA1[i,:]),  color=colors[i], linestyle ="dotted")
    #     plt.plot(10*np.log10(cent_dist_sama[i,:]), 10*np.log10(side_dist_sama[i,:]),  color=colors[i])
    #     plt.plot(10*np.log10(d0_list_theo), 10*np.log10(ds_list_theo), color = colors[i], linestyle="dashdot")
    # plt.xlabel(r"$D_0$")
    # plt.ylabel(r"$D_S$")
    # plt.show()
    
    
    # =============================================================================
    #     plots
    # =============================================================================
    ### IA vs Samara
    colors = sns.color_palette("bright")
    fig = plt.figure(3, figsize=(9, 6))
    plt.clf()
    plt.rc('font', size=14)
    plt.rc('figure', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=14)
    
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    #fig.suptitle("Rate vs. distortion")
    ax.set_xlabel(r"$D_0$ [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_ylabel(r"$D_S$ [dB]")
    
    # xlimit = [0, 10]
    
    line_styles = ["solid", "dashed", "dashdot"]
    for i, R in enumerate(rates):
        ds_list_theo, d0_list_theo = lower_bound_distortions(R, a, sig2w)
        
        ax.plot(10*np.log10(cent_dist_IA2[i,:]), 10*np.log10(side_dist_IA2[i,:]), color = colors[0], linestyle = line_styles[i], label="IA AR(2)" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(cent_dist_IA1[i,:]), 10*np.log10(side_dist_IA1[i,:]), color =  colors[1], linestyle = line_styles[i], label="IA AR(1)" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(cent_dist_sama[i,:]), 10*np.log10(side_dist_sama[i,:]), color =  colors[2], linestyle = line_styles[i], label="Sub-optimal staggering" if i ==0 else "_nolegend_")
        #ax.plot(10*np.log10(cent_dist_stag2[i,:]), 10*np.log10(side_dist_stag2[i,:]), color =  colors[3], linestyle = line_styles[i], label="Optimal staggering AR(2)" if i ==0 else "_nolegend_")
        #ax.plot(10*np.log10(cent_dist_stag1[i,:]), 10*np.log10(side_dist_stag1[i,:]), color =  colors[4], linestyle = line_styles[i], label="Optimal staggering AR(1)" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(d0_list_theo), 10*np.log10(ds_list_theo), color =  'k', linestyle = line_styles[i], label="Lower bound AR(1)" if i ==0 else "_nolegend_")
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=1, bbox_to_anchor=(0.11, 0.12))
    fig.tight_layout(rect=[0, 0.0, 1, 0.99])
    plt.savefig("theo_figs/side_vs_cent_IA.pdf")
    plt.show()
    
    
    ### samara vs optimal predictor 
    fig = plt.figure(4, figsize=(9, 6))
    plt.clf()
    plt.rc('font', size=14)
    plt.rc('figure', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=14)
    
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    #fig.suptitle("Rate vs. distortion")
    ax.set_xlabel(r"$D_0$ [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_ylabel(r"$D_S$ [dB]")
    
    # xlimit = [0, 10]
    
    line_styles = ["solid", "dashed", "dashdot"]
    for i, R in enumerate(rates):
        ds_list_theo, d0_list_theo = lower_bound_distortions(R, a, sig2w)
        
        #ax.plot(10*np.log10(cent_dist_IA2[i,:]), 10*np.log10(side_dist_IA2[i,:]), color = colors[0], linestyle = line_styles[i], label="IA AR(2)" if i ==0 else "_nolegend_")
        #ax.plot(10*np.log10(cent_dist_IA1[i,:]), 10*np.log10(side_dist_IA1[i,:]), color =  colors[1], linestyle = line_styles[i], label="IA AR(1)" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(cent_dist_sama[i,:]), 10*np.log10(side_dist_sama[i,:]), color =  colors[2], linestyle = line_styles[i], label="Sub-optimal staggering" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(cent_dist_stag2[i,:]), 10*np.log10(side_dist_stag2[i,:]), color =  colors[3], linestyle = line_styles[i], label="Optimal staggering AR(2)" if i ==0 else "_nolegend_")
        ax.plot(10*np.log10(cent_dist_stag1[i,:]), 10*np.log10(side_dist_stag1[i,:]), color =  colors[4], linestyle = line_styles[i], label="Optimal staggering AR(1)" if i ==0 else "_nolegend_")
        #ax.plot(10*np.log10(d0_list_theo), 10*np.log10(ds_list_theo), color =  'k', linestyle = line_styles[i], label="Lower bound AR(1)" if i ==0 else "_nolegend_")
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=1, bbox_to_anchor=(0.11, 0.18))
    fig.tight_layout(rect=[0, 0.0, 1, 0.99])
    plt.savefig("theo_figs/side_vs_cent_stag_w_rate_loss.pdf")
    plt.show()
    
    
    # #### D0 vs D0/Ds
    # fig = plt.figure(1, figsize=(9, 6))
    # plt.clf()
    # plt.rc('font', size=14)
    # plt.rc('figure', titlesize=14)
    # plt.rc('axes', labelsize=14)
    # plt.rc('axes', titlesize=14)
    
    # ax = fig.add_subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    
    # #fig.suptitle("Rate vs. distortion")
    # ax.set_xlabel(r"$D_0/D_S$ [dB]")
    # ax.minorticks_on()
    # ax.grid(alpha=0.5)
    # ax.grid(alpha=0.3, which='minor')
    
    # ax.set_ylabel(r"$D_0$ [dB]")
    
    # # xlimit = [0, 10]
    
    # line_styles = ["solid", "dashed", "dashdot"]
    # for i, R in enumerate(rates):
    #     ds_list_theo, d0_list_theo = lower_bound_distortions(R, a, sig2w)
        
    #     ax.plot(10*np.log10(cent_dist_IA2[i,:]/side_dist_IA2[i,:]), 10*np.log10(cent_dist_IA2[i,:]), color = colors[0], linestyle = line_styles[i], label="IA AR(2)" if i ==0 else "_nolegend_")
    #     ax.plot(10*np.log10(cent_dist_IA1[i,:]/side_dist_IA1[i,:]), 10*np.log10(cent_dist_IA1[i,:]), color =  colors[1], linestyle = line_styles[i], label="IA AR(1)" if i ==0 else "_nolegend_")
    #     ax.plot(10*np.log10(cent_dist_sama[i,:]/side_dist_sama[i,:]), 10*np.log10(cent_dist_sama[i,:]), color =  colors[2], linestyle = line_styles[i], label="Sub-optimal staggering" if i ==0 else "_nolegend_")
    #     #ax.plot(10*np.log10(cent_dist_stag2[i,:]), 10*np.log10(side_dist_stag2[i,:]), color =  colors[3], linestyle = line_styles[i], label="Optimal staggering AR(2)" if i ==0 else "_nolegend_")
    #     #ax.plot(10*np.log10(cent_dist_stag1[i,:]), 10*np.log10(side_dist_stag1[i,:]), color =  colors[4], linestyle = line_styles[i], label="Optimal staggering AR(1)" if i ==0 else "_nolegend_")
    #     ax.plot(10*np.log10(d0_list_theo/ds_list_theo), 10*np.log10(d0_list_theo), color =  'k', linestyle = line_styles[i], label="Lower bound AR(1)" if i ==0 else "_nolegend_")
    
    
    
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower left", ncol=1, bbox_to_anchor=(0.11, 0.18))
    # fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    # #plt.savefig(f"results/figures/rate_vs_distortion_{filenumber-1}.pdf")
    # plt.show()
    