# -*- coding: utf-8 -*-
"""
Created on Mon May 23 21:46:50 2022

@author: Kristian Søgaard

Plots used in the simulation chapter.
"""

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import lower_bound as lb
import seaborn as sns



def plot_results2(filenames):
    filename1 = filenames[0]
    filename2 = filenames[1]
    
    loadfile_IA = h5py.File(filename1, 'r')
    loadfile_stag = h5py.File(filename2, 'r')


        
    pars = loadfile_IA['Simulation Parameters']
    packet_loss_probs = pars['packet_loss_probs'][...]
    rates = pars['rates'][...]
    a = pars['a'][...]
    sig2w = pars['sig2w'][...]
    init_seed = pars['init_seed'][...]
    N_time = pars['N_time'][...] 
    M = pars['M'][...]
    nesting_ratios = pars['nesting_ratios'][...]
    
    #ar_1 parameters
    ar1_pars = loadfile_IA['AR1 Parameters']
    a_ar1 = ar1_pars['a'][...]
    sig2w_ar1 = ar1_pars['sig2w'][...]
    
    schemes = []
    schemes += ["IA AR_2"] 
    schemes += ["IA AR_1"]
    schemes += ["Samara" ]
    
    scheme_labels = []
    scheme_labels += [r"IA AR(2)" ]
    scheme_labels += [r"IA AR(1)" ]
    scheme_labels += [r"Sub-optimal stag" ]
    
    ### Distoriton vs rate different nesting ratios IA
    # fig = plt.figure(10, figsize=(9, 6))
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
    
    # fig.suptitle("Rate vs. distortion")
    # ax.set_xlabel(r"MSE distortion [dB]")
    # ax.minorticks_on()
    # ax.grid(alpha=0.5)
    # ax.grid(alpha=0.3, which='minor')
    
    # ax.set_ylabel(r"Rate")
    
    # # xlimit = [0, 10]
    
    # for i, scheme in enumerate(schemes[:2]):
    #     for j in range(len(nesting_ratios)):
    #         distortion = loadfile_IA[scheme]["D0"][...][1:,0,j]
    #         distortion = 10*np.log10(distortion)
    #         rates_op = loadfile_IA[scheme]['Operational_Rate'][...][1:,0,j]
        
        
    #         ax.plot(distortion, rates_op, label=scheme_labels[i] + fr", r = {nesting_ratios[j]}")
    
    
    
    # #     if scheme in ["IA AR_2", "IA AR_1", ]:
    # #         if np.amin(distortion) < xlimit[0]:
    # #             xlimit[0] = np.floor(np.amin(distortion)/5)*5 
    # #     else:
    # #         xlimit[1] = np.ceil(np.amax(distortion)/5)*5
        
        
    # # ax.set_xlim(xlimit)
    
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", ncol=1, bbox_to_anchor=(1, 0.9))
    # fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    # #plt.savefig(f"results/figures/rate_vs_distortion_{filenumber-1}.pdf")
    # plt.show()
    
    # ### Distoriton vs rate different rateconfigs Samara
    # fig = plt.figure(2, figsize=(9, 6))
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
    
    # fig.suptitle("Rate vs. distortion")
    # ax.set_xlabel(r"MSE distortion [dB]")
    # ax.minorticks_on()
    # ax.grid(alpha=0.5)
    # ax.grid(alpha=0.3, which='minor')
    
    # ax.set_ylabel(r"Rate")
    
    # # xlimit = [0, 10]
    
    # scheme = schemes[2]
    # for j in range(10):
    #     distortion = loadfile[scheme]["D0"][...][1:,0,j]
    #     distortion = 10*np.log10(distortion)
    #     rates_op = loadfile[scheme]['Operational_Rate'][...][1:,0,j]
    
    
    #     ax.plot(distortion, rates_op, label=scheme_labels[2] + fr", r = {j}")


    # rates = np.linspace(5,12, 50)
    # theo_bound = lb.DRF(rates/2, a_ar1[0], sig2w_ar1)

    # ax.plot(10*np.log10(theo_bound), rates, label="Theoretic AR(1)")
    # #     if scheme in ["IA AR_2", "IA AR_1", ]:
    # #         if np.amin(distortion) < xlimit[0]:
    # #             xlimit[0] = np.floor(np.amin(distortion)/5)*5 
    # #     else:
    # #         xlimit[1] = np.ceil(np.amax(distortion)/5)*5
        
        
    # # ax.set_xlim(xlimit)
    
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", ncol=1, bbox_to_anchor=(1, 0.9))
    # fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    # #plt.savefig(f"results/figures/rate_vs_distortion_{filenumber-1}.pdf")
    # plt.show()
    
    
    ### Distoriton vs rate IA nesting_ratio = 7, best Samara
    colors = sns.color_palette("bright")
    linestyles = ["solid", "dashed", "dashdot"]
    markers = ["o", 'v', '^']
    
    fig = plt.figure(1, figsize=(9, 6))
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
    ax.set_xlabel(r"MSE distortion [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_ylabel(r"Sum-rate")
    
    # xlimit = [0, 10]
    
    for i, scheme in enumerate(schemes[:2]):
        distortion = loadfile_IA[scheme]["D0"][...][:,0,2]
        distortion = 10*np.log10(distortion)
        rates_op = loadfile_IA[scheme]['Operational_Rate'][...][:,0,2]
    
    
        ax.plot(distortion, rates_op, label=scheme_labels[i], color=colors[i], marker = markers[i], markersize=5)
    
    scheme = schemes[2]
    distortions = loadfile_stag[scheme]["D0"][...][:,0,:]
    distortions[distortions == 0] = distortions.max()
    idx =  np.argmin(distortions, axis = 1)
    distortion = distortions[np.arange(distortions.shape[0]), idx]
    distortion = 10*np.log10(distortion)
    rates_op = loadfile_stag[scheme]['Operational_Rate'][...][:,0,:]
    rates_op = rates_op[np.arange(rates_op.shape[0]), idx]


    ax.plot(distortion, rates_op, label=scheme_labels[2], color=colors[2], marker = markers[2], markersize=5)
    
    rates = np.linspace(5,12, 50)
    theo_bound = lb.DRF(rates/2, a_ar1[0], sig2w_ar1)

    ax.plot(10*np.log10(theo_bound), rates, label="Theoretic AR(1)", color="k")
    #     if scheme in ["IA AR_2", "IA AR_1", ]:
    #         if np.amin(distortion) < xlimit[0]:
    #             xlimit[0] = np.floor(np.amin(distortion)/5)*5 
    #     else:
    #         xlimit[1] = np.ceil(np.amax(distortion)/5)*5
        
        
    # ax.set_xlim(xlimit)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=1, bbox_to_anchor=(1, 0.9))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    #plt.savefig(f"results/figures/rate_vs_distortion_central.pdf")
    plt.show()
    
    
    ### Side Distoriton vs rate IA nesting_ratio = 7, best Samara
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
    
    fig.suptitle("Rate vs. distortion")
    ax.set_xlabel(r"MSE distortion [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_ylabel(r"Rate")
    
    # xlimit = [0, 10]
    
    for i, scheme in enumerate(schemes[:2]):
        distortion = loadfile_IA[scheme]["Ds"][...][:,0,2]
        distortion = 10*np.log10(distortion)
        rates_op = loadfile_IA[scheme]['Operational_Rate'][...][:,0,2]
    
    
        ax.plot(distortion, rates_op, label=scheme_labels[i])
    
    scheme = schemes[2]
    distortions = loadfile_stag[scheme]["D0"][...][:,0,:]
    distortions[distortions == 0] = distortions.max()
    idx =  np.argmin(distortions, axis = 1)
    distortion = loadfile_stag[scheme]["Ds"][...][:,0,:]
    distortion = distortion[np.arange(distortion.shape[0]), idx]
    distortion = 10*np.log10(distortion)
    rates_op = loadfile_stag[scheme]['Operational_Rate'][...][:,0,:]
    rates_op = rates_op[np.arange(rates_op.shape[0]), idx]


    ax.plot(distortion, rates_op, label=scheme_labels[2])
    
    # rates = np.linspace(5,12, 50)
    # theo_bound = lb.DRF(rates/2, a_ar1[0], sig2w_ar1)

    # ax.plot(10*np.log10(theo_bound), rates, label="Theoretic AR(1)")
    #     if scheme in ["IA AR_2", "IA AR_1", ]:
    #         if np.amin(distortion) < xlimit[0]:
    #             xlimit[0] = np.floor(np.amin(distortion)/5)*5 
    #     else:
    #         xlimit[1] = np.ceil(np.amax(distortion)/5)*5
        
        
    # ax.set_xlim(xlimit)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=1, bbox_to_anchor=(1, 0.9))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    #plt.savefig(f"results/figures/rate_vs_distortion_{filenumber-1}.pdf")
    plt.show()
    
    
    # =============================================================================
    #   Distortion vs packetloss
    # =============================================================================
    
    ### Distortion vs packetloss IA only
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
    
    fig.suptitle("Distortion vs. packet loss probability")
    ax.set_ylabel(r"MSE distortion [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_xlabel(r"Packet loss probability %")
    
    rate_for_theo = []
    for i, scheme in enumerate(schemes[:1]):
        for j in range(len(nesting_ratios)):
            rate_op = 10
            rates_op = np.mean(loadfile_IA[scheme]['Operational_Rate'][...][:,:,j], axis=1)
            #print(rates_op)
            #rates_op = rates_op[rates_op >=6]
            
            idx = (np.abs(rates_op - rate_op)).argmin()
            rate_op = rates_op[idx]
            rate_for_theo.append(rate_op)
            
            distortion = loadfile_IA[scheme]["D0"][...]
            distortion = loadfile_IA[scheme]["D0"][...][idx,:,j]
            distortion = 10*np.log10(distortion)
            
            ax.plot(packet_loss_probs, distortion, label=scheme_labels[i] + fr" $R={np.round(rate_op, decimals=2)}$, $r = {nesting_ratios[j]}$")
    
    
    
    #Theoretic
    theo_bound = lb.lower_bound_packet_loss(np.mean(rate_for_theo)/2, a_ar1[0], sig2w_ar1, packet_loss_probs)
    ax.plot(packet_loss_probs, 10*np.log10(theo_bound), label=fr"Theoretic $R={np.round(np.mean(rate_for_theo), decimals=2)}$")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=1, bbox_to_anchor=(1, 0.18))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    #plt.savefig(f"results/figures/distortion_vs_plp_{filenumber-1}.pdf")
    plt.show()
    
    
    ### Distortion vs packetloss Best IA and Samara
    fig = plt.figure(5, figsize=(9, 6))
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
    
    # fig.suptitle("Distortion vs. packet loss probability")
    ax.set_ylabel(r"MSE distortion [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_xlabel(r"Packet loss probability %")
    packet_loss_probs = packet_loss_probs*100
    rate_for_theo = []
    distortions = []
    for i, scheme in enumerate(schemes[:2]):
        rate_op = 10
        rates_op = np.mean(loadfile_IA[scheme]['Operational_Rate'][...][:,:,:], axis=1)
        #print(rates_op)
        #rates_op = rates_op[rates_op >=6]
        
        idx = (np.abs(rates_op - rate_op)).argmin(axis=0)
        rate_op = rates_op[idx, np.arange(rates_op.shape[1])]
        
        distortion = loadfile_IA[scheme]["D0"][...][:,:,:]
        weigh_idx = np.argmin(distortion[idx,:,np.arange(rates_op.shape[1])], axis=0)
        distortion = np.min(distortion[idx,:,np.arange(rates_op.shape[1])], axis=0)
        #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
        distortion = 10*np.log10(distortion)
        distortions.append(distortion)
        #weigthed_rate
        #weigh_idx = np.argmin(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
        idx, counts = np.unique(weigh_idx, return_counts=True)
        rate_op = sum([rate_op[idx[i]]*counts[i]/np.sum(counts) for i in range(len(idx))])
        rate_for_theo.append(rate_op)
        
        ax.plot(packet_loss_probs, distortion, label=scheme_labels[i] + fr" $R={np.round(rate_op, decimals=2)}$", color=colors[i], marker = markers[i], markersize=5)
    
    scheme = schemes[2]
    rate_op = 10
    rates_op = np.mean(loadfile_stag[scheme]['Operational_Rate'][...][:,:,:], axis=1)
    #print(rates_op)
    #rates_op = rates_op[rates_op >=6]
    
    idx = (np.abs(rates_op - rate_op)).argmin(axis=0)
    rate_op = rates_op[idx, np.arange(rates_op.shape[1])]
    
    distortion = loadfile_stag[scheme]["D0"][...][:,:,:]
    distortion[distortion == 0] = distortion.max()
    weigh_idx = np.argmin(distortion[idx,:,np.arange(rates_op.shape[1])], axis=0)
    distortion = np.min(distortion[idx,:,np.arange(rates_op.shape[1])], axis=0)
    #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
    distortion = 10*np.log10(distortion)
    distortions.append(distortion)
    #weigthed_rate
    #weigh_idx = np.argmin(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
    idx, counts = np.unique(weigh_idx, return_counts=True)
    rate_op = sum([rate_op[idx[i]]*counts[i]/np.sum(counts) for i in range(len(idx))])
    rate_for_theo.append(rate_op)
    
    ax.plot(packet_loss_probs, distortion, label=scheme_labels[i] + fr" $R={np.round(rate_op, decimals=2)}$", color=colors[2], marker = markers[2], markersize=5)
    
    #Theoretic
    packet_loss_probs = packet_loss_probs/100
    theo_bound = lb.lower_bound_packet_loss(np.mean(rate_for_theo)/2, a_ar1[0], sig2w_ar1, packet_loss_probs)
    packet_loss_probs = packet_loss_probs*100
    ax.plot(packet_loss_probs, 10*np.log10(theo_bound), label=fr"Theoretic $R={np.round(np.mean(rate_for_theo), decimals=2)}$", color="k")
    # theo_bound = lb.lower_bound_packet_loss(8.23/2, a_ar1[0], sig2w_ar1, packet_loss_probs)
    # ax.plot(packet_loss_probs, 10*np.log10(theo_bound), label=fr"Theoretic $R={np.round(8.23, decimals=2)}$")
    distortions.append(10*np.log10(theo_bound))
    ax.set_ylim(-40,-20)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=1, bbox_to_anchor=(1, 0.12))
    fig.tight_layout(rect=[0, 0, 1, 1])
    #plt.savefig(f"results/figures/distortion_vs_plp_zoom.pdf")
    plt.show()
    
    
    # =============================================================================
    #    Central vs side distortion
    # =============================================================================
        
    fig = plt.figure(6, figsize=(9, 5.5))
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
    
    #vfig.suptitle("Central distortion vs. Side distortion")
    ax.set_ylabel(r"$D_S$ [dB]")
    ax.minorticks_on()
    ax.grid(alpha=0.5)
    ax.grid(alpha=0.3, which='minor')
    
    ax.set_xlabel(r"$D_0$ [dB]")
    ax.set_xlim(-66, -33)
    
    colors = sns.color_palette("bright")
    linestyles = ["solid", "dashed", "dashdot"]
    markers = ["o", 'v', '^']
    for j, rate_op_theo in enumerate([8, 10, 12]):
        rate_for_theo = []
        for i, scheme in enumerate(schemes[:2]):
            
            rate_op = rate_op_theo
            rates_op = loadfile_IA[scheme]['Operational_Rate'][...][:,0,:]
            #print(rates_op)
            idxs = np.mean(rates_op, axis=1) >= rate_op
            rates_op = rates_op[idxs,:]
            
            # idx = (np.abs(rates_op - rate_op)).argmin(axis=0)
            # rate_op = rates_op[idx, np.arange(rates_op.shape[1])]
            
            # D0 = loadfile[scheme]["D0"][...][1:,0,:]
            # D0 = D0[idx, np.arange(rates_op.shape[1])]
            # #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
            # D0 = 10*np.log10(D0)
            
            # DS = loadfile[scheme]["Ds"][...][1:,0,:]
            # DS = DS[idx, np.arange(rates_op.shape[1])]
            # #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
            # DS = 10*np.log10(DS)
            
            # rate_op = np.mean(rate_op)
            # rate_for_theo.append(rate_op)
            
            # ax.plot(D0, DS, label=scheme_labels[i] + fr" $R={np.round(rate_op, decimals=2)}$")
            
            
            idx = (np.abs(np.mean(rates_op, axis=1) - rate_op)).argmin()
            rate_op = rates_op[idx, :]
            D0 = loadfile_IA[scheme]["D0"][...][idxs,0,:]
            D0 = 10*np.log10(D0[idx,:])
            DS = loadfile_IA[scheme]["Ds"][...][idxs,0,:]
            DS = 10*np.log10(DS[idx,:])
            rate_op = np.mean(rate_op)
            rate_for_theo.append(rate_op)
            ax.plot(D0, DS, label=scheme_labels[i] + fr",{14*' '} $R={np.round(rate_op, decimals=2)}$", color=colors[i], linestyle=linestyles[j], marker = markers[i], markersize=5)
        
        scheme = schemes[2]
        rate_op = rate_op_theo
        rates_op = loadfile_stag[scheme]['Operational_Rate'][...][:,0,:]
        #print(rates_op)
        idxs = np.mean(rates_op, axis=1) >= rate_op
        rates_op = rates_op[idxs,:]
        
        # idx = (np.abs(rates_op - rate_op)).argmin(axis=0)
        # rate_op = rates_op[idx, np.arange(rates_op.shape[1])]
        
        # D0 = loadfile[scheme]["D0"][...][1:,0,:]
        # D0 = D0[idx, np.arange(rates_op.shape[1])]
        # #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
        # D0 = 10*np.log10(D0)
        
        # DS = loadfile[scheme]["Ds"][...][1:,0,:]
        # DS = DS[idx, np.arange(rates_op.shape[1])]
        # #distortion = np.min(loadfile[scheme]["D0"][...][idx,:,np.arange(rates_op.shape[1])], axis=0)
        # DS = 10*np.log10(DS)
        
        # rate_op = np.mean(rate_op)
        # rate_for_theo.append(rate_op)
        
        # ax.plot(D0, DS, label=scheme_labels[i] + fr" $R={np.round(rate_op, decimals=2)}$")
        
        
        idx = (np.abs(np.mean(rates_op, axis=1) - rate_op)).argmin()
        rate_op = rates_op[idx, :]
        D0 = loadfile_stag[scheme]["D0"][...][idxs,0,:]
        D0 = 10*np.log10(D0[idx,:])
        DS = loadfile_stag[scheme]["Ds"][...][idxs,0,:]
        DS = 10*np.log10(DS[idx,:])
        rate_op = np.mean(rate_op)
        rate_for_theo.append(rate_op)
        ax.plot(D0, DS, label=scheme_labels[2] + fr", $R={np.round(rate_op, decimals=2)}$", color=colors[2], linestyle=linestyles[j], marker = markers[2], markersize=5)
    
            
        
        d0_theo, ds_theo, _ = lb.theo_dist_bound((rate_for_theo[0])/2, a_ar1, sig2w_ar1, N=1000, Ds_bound=0, fix_rho=True)
        ax.plot(10*np.log10(d0_theo), 10*np.log10(ds_theo), label=fr"Theoretic,{12*' '} $R={np.round((rate_for_theo[0]), decimals=2)}$", color="k", linestyle=linestyles[j], zorder=1)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", ncol=1, bbox_to_anchor=(1, 0.55))
    fig.tight_layout(rect=[0, 0, 0.65, 1])
    #plt.savefig(f"results/figures/side_vs_cent_simulation.pdf")
    plt.show()


if __name__ == "__main__":
    
    # filename = "results/simulated_performance_of_schemes46.hdf5"
    # plot_results(filename = "results/simulated_performance_of_schemes46.hdf5")
    
    filenames = ["results/simulated_performance_of_schemes46.hdf5", "results/simulated_performance_of_schemes_stag52.hdf5"]
    plot_results2(filenames)
    
    
    low = lambda ds: 2*ds-0.126316
    up = lambda ds: 1/(2/ds - 1/0.126316)
    
    