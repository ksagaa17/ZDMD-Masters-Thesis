# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:59:05 2022

@author: Kristian Søgaard

Script for simulation of the MSE performance of the ZDMD schemes
"""

import numpy as np
import feed_back_index_assignment as feedback_IA
import entropy_coder as ec
import sama_scheme as sama
import stag_scheme as stag
import multiprocessing as mp
import time
import os
import h5py
import AR_process as AR



def rate_distortion_IA_packetloss(X, R, a, sig2w, packet_loss_prob):
    nesting_ratios = [3, 5, 7]
    D0 = np.zeros(len(nesting_ratios))
    Ds = np.zeros(len(nesting_ratios))
    sum_rate = np.zeros(len(nesting_ratios))
    avg_rate = np.zeros(len(nesting_ratios))
    
    for i, nesting_ratio in enumerate(nesting_ratios):
        index_map = np.genfromtxt(f"tables/IA_matrix_{nesting_ratio}_2.csv", delimiter=',')
        
        Delta_C = feedback_IA.binsize_index_assignment(R, a, sig2w, r=nesting_ratio)
        #Delta_C = np.round(Delta_C, decimals = 7)
        q, Y, U, Y_side = feedback_IA.run_index_assigment(X, Delta_C, index_map, a, packet_loss_prob)
        
        D0[i] = np.mean((X-Y)**2)
        Ds[i] = np.mean(0.5*(X - Y_side[:,0])**2 + 0.5*(X - Y_side[:,1])**2)
        
        ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(q[2:, 0], decimals=12), len(q[2:, 0]))
        ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(q[2:, 1], decimals=12), len(q[2:, 1]))
        
        sum_rate[i] = bitrate_1 + bitrate_2
        avg_rate[i] = sum_rate[i]/2
    
    return D0, Ds, sum_rate, avg_rate

    

def rate_distortion_sama_packetloss(X, R, a, sig2w, packet_loss_prob, print_rates= False):
    R0_list = np.linspace(0, R-3, 10)
    R1_list = (R - R0_list)/2
    
    D0 = np.zeros(len(R0_list))
    Ds = np.zeros(len(R0_list))
    sum_rate = np.zeros(len(R0_list))
    avg_rate = np.zeros(len(R0_list))
    
    R0_list = R0_list[R1_list > 1]
    R1_list = R1_list[R1_list > 1]
    
    
    for i, R0 in enumerate(R0_list):
        R1 = R1_list[i]
        U_quant, e_c_quant, Y_0, Y_side, Y_used = sama.run_samacheme(X, R1, R0, a, sig2w, packet_loss_prob)

        D0[i] = np.mean((X-Y_used)**2)
        Ds[i] = np.mean(0.5*(X - Y_side[:,0])**2 + 0.5*(X - Y_side[:,1])**2)
        
        ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(U_quant[2:, 0], decimals=12), len(U_quant[2:, 0]))
        ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(U_quant[2:, 1], decimals=12), len(U_quant[2:, 1]))
        ent_3, avg_rate_3, bitrate_3 = ec.entropy_coder(np.round(e_c_quant[2:], decimals=12), len(e_c_quant[2:]))
        
        if print_rates:
            print_one_pair(R1, R0, bitrate_1, bitrate_2, bitrate_3, ent_1, ent_2, ent_3)
            
        sum_rate[i] = bitrate_1 + bitrate_2 + bitrate_3
        avg_rate[i] = sum_rate[i]/2
    
    return D0, Ds, sum_rate, avg_rate, R0_list, R1_list


def rate_distortion_stag_packetloss(X, R, a, sig2w, packet_loss_prob, print_rates= False):
    R0_list = np.linspace(0, R-3, 10)
    R1_list = (R - R0_list)/2
    
    D0 = np.zeros(len(R0_list))
    Ds = np.zeros(len(R0_list))
    sum_rate = np.zeros(len(R0_list))
    avg_rate = np.zeros(len(R0_list))
    
    R0_list = R0_list[R1_list > 1]
    R1_list = R1_list[R1_list > 1]
    
    
    for i, R0 in enumerate(R0_list):
        R1 = R1_list[i]
        U_quant, e_c_quant, Y_0, Y_side, Y_used = stag.run_stag_scheme(X, R1, R0, a, sig2w, packet_loss_prob)

        D0[i] = np.mean((X-Y_used)**2)
        Ds[i] = np.mean(0.5*(X - Y_side[:,0])**2 + 0.5*(X - Y_side[:,1])**2)
        
        ent_1, avg_rate_1, bitrate_1 = ec.entropy_coder(np.round(U_quant[2:, 0], decimals=12), len(U_quant[2:, 0]))
        ent_2, avg_rate_2, bitrate_2 = ec.entropy_coder(np.round(U_quant[2:, 1], decimals=12), len(U_quant[2:, 1]))
        ent_3, avg_rate_3, bitrate_3 = ec.entropy_coder(np.round(e_c_quant[2:], decimals=12), len(e_c_quant[2:]))
        
        if print_rates:
            print_one_pair(R1, R0, bitrate_1, bitrate_2, bitrate_3, ent_1, ent_2, ent_3)
            
        # sum_rate[i] = bitrate_1 + bitrate_2 + bitrate_3
        sum_rate[i] = 2*bitrate_1 + bitrate_3 + 1
        avg_rate[i] = sum_rate[i]/2
    
    return D0, Ds, sum_rate, avg_rate, R0_list, R1_list


def print_one_pair(R1, R0, R1_op, R2_op, R3_op, Q1_ent, Q2_ent, Q3_ent):
    print("Rates:")
    print("Desired R1={:3.3f}, R1_op={:.5f}, Q1_ent={:.5f}".format(R1, R1_op, Q1_ent))
    print("Desired R2={:3.3f}, R2_op={:.5f}, Q2_ent={:.5f}".format(R1, R2_op, Q2_ent))
    print("Desired R0={:3.3f}, R0_op={:.5f}, Q0_ent={:.5f}".format(R0, R3_op, Q3_ent))
    R_sum_op = (R1_op + R2_op + R3_op)
    print("Desired avg. sum rate={}. Rs_op={:.5f}".format(R0+2*R1, R_sum_op))


#### MP SETUP #####
def mp_run_scheme(X, rates, a, sig2w, packet_loss_probs, scheme, j=0):
    archives = {}
    # Ensure each process has different seed each run
    mpid = mp.current_process()._identity[0]
    try:
        mpseed = mpid * int(1E8*(time.time()-int(time.time())))
        np.random.seed(seed=mpseed)
    except ValueError:

        while mpseed > 2**32:
            mpseed = int(mpid * int(1E8*(time.time()-int(time.time())))/np.random.randint(1, 4))
        np.random.seed(seed=mpseed)
    # finally:
    #     print(mpseed)
    
    if scheme == rate_distortion_IA_packetloss:
        points = 3
        archives['D0'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Ds'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Operational_Rate'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Desired_Rate'] = np.zeros((len(rates), len(packet_loss_probs)))
        
    else:
        points = 10
        archives['D0'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Ds'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Operational_Rate'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['Desired_Rate'] = np.zeros((len(rates), len(packet_loss_probs)))
        archives['R0'] = np.zeros((len(rates), len(packet_loss_probs), points))
        archives['R1'] = np.zeros((len(rates), len(packet_loss_probs), points))
    
    
    for i, rate in enumerate(rates):
        for k, packet_loss_prob in enumerate(packet_loss_probs):
            print(rate, "  ", packet_loss_prob)
            input_vars = X, rate, a, sig2w, packet_loss_prob
            res = scheme(*input_vars)
            
            if len(res) == 4:
                archives['D0'][i,k,:] = res[0]
                archives['Ds'][i,k,:] = res[1]
                archives['Operational_Rate'][i,k,:] = res[2]
                
            else:
                archives['D0'][i,k,:] = res[0]
                archives['Ds'][i,k,:] = res[1]
                archives['Operational_Rate'][i,k,:] = res[2]
                archives['R0'][i,k,:len(res[4])] = res[4]
                archives['R1'][i,k,:len(res[5])] = res[5]
                
        archives['Desired_Rate'][i,:] = rate*np.ones(len(packet_loss_probs))
    
    pid = os.getpid()
    print("Process {:d} finished sim # {:d}".format(pid, j))
    return archives


def parallel_simulation(scheme, M_p, M_monte, X, rates, a, sig2w, packet_loss_probs):
    """
    Evt. kræve at X.shape[1] = M_monte
    """
    pool = mp.Pool(M_p)
    input_vars = (rates, a, sig2w, packet_loss_probs, scheme)
    mp_input = [(X[j,:],) + input_vars + (j,)for j in range(M_monte)]
    mp_res = pool.starmap(mp_run_scheme, mp_input)
    pool.close()
    pool.join()
    return mp_res

### Simulations
def simulate_one_scheme(scheme, processors, M, X, rates, a, sig2w, packet_loss_probs, save_file, group):
    t1 = time.time()
    mp_res = parallel_simulation(scheme, processors, M, X, rates, a, sig2w, packet_loss_probs)
    t2 = time.time()
    print("Total time: "+ f"{(t2-t1)//60:02}:{(t2-t1)%60:02}")
    ### Get means and save results ####
    archives = {}
    for key in mp_res[0].keys():
        shape = mp_res[0][key].shape
        
        archives[key] = np.zeros(((M, ) + shape))

    for key in archives.keys():
        for j in range(M):
            archives[key][j] = mp_res[j][key]
        archives[key] = np.mean(archives[key], axis=0)

    
    try:
        result_grp = save_file.create_group(group)
    except:
        result_grp = save_file[group]
        
    for key in archives.keys():
        try:
            result_grp.create_dataset(key, data=archives[key])
        except:
            result_grp[key][...] = archives[key]
        
    return archives


    

def simulate_performance(a, sig2w, N_time, init_seed):
    np.random.seed(init_seed)
    M = 4
    r = AR.get_acf_from_params(a, sig2w)
    X = np.array([AR.generate_AR_process(r, N_time)[0] for i in range(M)])
    rates = np.linspace(5,12,15)
    packet_loss_probs = np.linspace(0,0.24, 25)
    nesting_ratios = [3, 5, 7]
    
    
    
    filenumber = len([name for name in os.listdir('./results') if name[-5:] == ".hdf5"])
    filename = f"results/simulated_performance_of_schemes{filenumber}.hdf5"
    save_file = h5py.File(filename, 'a')
    
    
    # Save Parameters #
    pars = {}
    pars['a'] = a
    pars['sig2w'] = sig2w
    pars['init_seed'] = init_seed
    pars['N_time'] = N_time
    pars['M'] = M
    pars['rates'] = rates
    pars['packet_loss_probs'] = packet_loss_probs
    pars['Source'] = X
    pars['nesting_ratios'] = nesting_ratios
    try:
        pars_grp = save_file.create_group("Simulation Parameters")
    except:
        pars_grp = save_file["Simulation Parameters"]

    for key in pars.keys():
        try:
            pars_grp.create_dataset(key, data=pars[key])
        except:
            pars_grp[key][...] = pars[key]
    
    
        
        
    group = "IA AR_2"
    archives_ar2 = simulate_one_scheme(rate_distortion_IA_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)
    
    # group = f"IA AR_2 no feedback"
    # archives_ar2 = simulate_one_scheme(rate_distortion_IA_no_feedback, 4, M, X, rates, a, sig2w, 
    #                     packet_loss_probs, save_file, group)
    
    a, sig2w = AR.find_params(r[:2])
    
    pars = {}
    pars['a'] = a
    pars['sig2w'] = sig2w
    try:
        pars_grp = save_file.create_group("AR1 Parameters")
    except:
        pars_grp = save_file["AR1 Parameters"]

    for key in pars.keys():
        try:
            pars_grp.create_dataset(key, data=pars[key])
        except:
            pars_grp[key][...] = pars[key]
    
    group = "IA AR_1"
    archives_ar1 = simulate_one_scheme(rate_distortion_IA_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)

    
    group = "Samara"
    archives_sama = simulate_one_scheme(rate_distortion_sama_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)
    
    save_file.close()
    
    return archives_ar2, archives_ar1, archives_sama


def simulate_stag_performance(a, sig2w, N_time, init_seed):
    np.random.seed(init_seed)
    M = 4
    r = AR.get_acf_from_params(a, sig2w)
    X = np.array([AR.generate_AR_process(r, N_time)[0] for i in range(M)])
    rates = np.linspace(5,12,15)
    packet_loss_probs = np.linspace(0,0.24, 25)
    nesting_ratios = [3, 5, 7]
    
    
    
    filenumber = len([name for name in os.listdir('./results') if name[-5:] == ".hdf5"])
    filename = f"results/simulated_performance_of_schemes_stag{filenumber}.hdf5"
    save_file = h5py.File(filename, 'a')
    
    
    # Save Parameters #
    pars = {}
    pars['a'] = a
    pars['sig2w'] = sig2w
    pars['init_seed'] = init_seed
    pars['N_time'] = N_time
    pars['M'] = M
    pars['rates'] = rates
    pars['packet_loss_probs'] = packet_loss_probs
    pars['Source'] = X
    pars['nesting_ratios'] = nesting_ratios
    try:
        pars_grp = save_file.create_group("Simulation Parameters")
    except:
        pars_grp = save_file["Simulation Parameters"]

    for key in pars.keys():
        try:
            pars_grp.create_dataset(key, data=pars[key])
        except:
            pars_grp[key][...] = pars[key]
    
    
        
        
    group = "Stag AR_2"
    archives_ar2 = simulate_one_scheme(rate_distortion_stag_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)
    
    
    a, sig2w = AR.find_params(r[:2])
    
    pars = {}
    pars['a'] = a
    pars['sig2w'] = sig2w
    try:
        pars_grp = save_file.create_group("AR1 Parameters")
    except:
        pars_grp = save_file["AR1 Parameters"]

    for key in pars.keys():
        try:
            pars_grp.create_dataset(key, data=pars[key])
        except:
            pars_grp[key][...] = pars[key]
    
    group = "Stag AR_1"
    archives_ar1 = simulate_one_scheme(rate_distortion_stag_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)

    group = "Samara"
    archives_sama = simulate_one_scheme(rate_distortion_sama_packetloss, 4, M, X, rates, a, sig2w, 
                        packet_loss_probs, save_file, group)
    save_file.close()
    
    return archives_ar2, archives_ar1, archives_sama


def test_simulation(a, sig2w, N_time, init_seed):
    np.random.seed(init_seed)
    
    r = AR.get_acf_from_params(a, sig2w)
    X = AR.generate_AR_process(r, N_time)[0]
    rates = np.linspace(5,12,15)
    packet_loss_probs = [0]
    
    a, sig2w = AR.find_params(r[:2])
    
    D0 = np.zeros((15, 10))
    Ds = np.zeros((15, 10))
    sum_rate = np.zeros((15, 10))
    avg_rate = np.zeros((15, 10))
    R0_list = np.zeros((15, 10))
    R1_list = np.zeros((15, 10))
    for i, rate in enumerate(rates):
        print(i)
        D0[i,:], Ds[i,:], sum_rate[i,:], avg_rate[i,:], R0_list[i,:], R1_list[i,:] = rate_distortion_sama_packetloss(X, rate, a, sig2w, packet_loss_probs[0], print_rates= False)


if __name__ == "__main__":
    
    
    
    # Parameters #
    init_seed = 123456
    a = np.array([ 1.42105263, -0.57894737])
    sig2w = 0.126316
    N_time = 100000
    
    # archives_ar2, archives_ar1, archives_sama = simulate_performance(a, sig2w, N_time, init_seed)
    archives_ar2, archives_ar1, archives_sama = simulate_stag_performance(a, sig2w, N_time, init_seed)
    
    #plot_results(filename = None)
    
    
    
    
    
    
    
    
    
    
    
    
    
  
        
        