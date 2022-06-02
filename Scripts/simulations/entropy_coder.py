# -*- coding: utf-8 -*-
"""
Author: Andreas J. Fuglsig

Entropy coder and operational rates calculations
"""
import numpy as np
import huffman as huff


### Entropy coder ###
def entropy_coder(mat, N):
    """
    Estimate the entropy, expected code length, and perform Huffman
    coding of a signal, mat, of size (N, 1)
    """
    # Get unique quantized values and number of occurrences pr. bin
    bins, counts = np.unique(mat, axis=0, return_counts=True)
    no_bins = len(counts)
    if no_bins > 2:
        # Estimate pmf
        pn = counts/N
        # Estimate entropy
        ent = -np.sum(pn*np.log2(pn))

        code = huff.huffman2(pn)
        length = huff.get_code_length(code)
        avg_rate = np.sum(pn*length)  # Expected code length
        diff = abs(mat[:, np.newaxis] - bins)**2  # Only for scalar case
        val = np.min(diff, axis=1)
        if val.sum() != 0:
            raise ValueError("Something was not coded")
        # Codeword index for each signal sample
        index = np.argmin(diff, axis=1)
        # Assign codeword to each symbol and measure operational rate
        bitrate = np.sum(length[index]) / N
    else:
        bitrate = 0
        avg_rate = 0
        ent = 0
    return ent, avg_rate, bitrate


def operatinal_rates(zeta_quant, e_c_quant, N):
    Q1_ent, Q1_avg_rate, R1_op = entropy_coder(zeta_quant[1:, 0], N-1)
    Q2_ent, Q2_avg_rate, R2_op = entropy_coder(zeta_quant[1:, 1], N-1)
    Q3_ent, Q3_avg_rate, R3_op = entropy_coder(e_c_quant[1:], N-1)

    Rs_op = (R1_op + R2_op + R3_op)/2
    return Rs_op, R1_op, R2_op

if __name__ == "__main__":
    N = 10000
    X = np.random.randn(N)
    Delta = 0.5
    
    Q = np.round(X/Delta)*Delta
    
    ent, avg_rate, bitrate = entropy_coder(Q, N)

