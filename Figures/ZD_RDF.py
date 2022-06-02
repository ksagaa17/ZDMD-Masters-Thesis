# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:24:49 2021

@author: krist
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


a = 0.7
sig2x = 1
sig2w = (1-a**2)*sig2x
D = np.linspace(0,1,1000)[1:-1]

R = 0.5*np.log2((sig2w*(1-a**2))/D)
R_ZD = 0.5*np.log2(sig2w/D + a**2)

font_size = 12

colors = sns.color_palette("bright")
fig = plt.figure(1, figsize=(5, 3.5))
plt.clf()
plt.rc('font', size=font_size)
plt.rc('figure', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('axes', titlesize=font_size)

ax = fig.add_subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

ax.set(xlabel=r'$D$', ylabel=r'$R$')
ax.minorticks_on()
ax.grid(alpha=0.5)
ax.grid(alpha=0.3, which='minor')

ax.plot(D,R, color=colors[0], label=r'$R(D)$')
ax.plot(D, R_ZD, color=colors[1], label=r'$R_{ZD}^I(D)$')

ax.set_ylim(0,4)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=1, bbox_to_anchor=(1, 1))
fig.tight_layout(rect=[0, 0.0, 1, 0.99])
# plt.savefig("ZD_rates.pdf")
plt.show()



