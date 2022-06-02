# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:24:49 2021

@author: krist
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## example 1
D_S = np.linspace(0,1,1000)[1:-1]
D_C = D_S/(2-D_S)
low = D_S**2

font_size = 12

colors = sns.color_palette("bright")
fig = plt.figure(1, figsize=(5, 4))
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

ax.set(xlabel=r'$D_S$ [dB]', ylabel=r'$D_0$ [dB]')
ax.minorticks_on()
ax.grid(alpha=0.5)
ax.grid(alpha=0.3, which='minor')

ax.plot(10*np.log10(D_S),10*np.log10(D_C), color=colors[0], label=r'$D_S/(2-D_S)$')
ax.plot(10*np.log10(D_S), 10*np.log10(low), color=colors[1], label=r'$D_S^2$')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", ncol=1, bbox_to_anchor=(1, 0.2))
fig.tight_layout(rect=[0, 0.0, 1, 0.99])
# plt.savefig("MD_dist_bound_exa_1.pdf")
plt.show()



## example 2
D_1 = np.linspace(0,1,10000000)[1:-1]
R_1 = 4
R_2 = 4

D_2 = 1 + np.exp(-2*(R_1+R_2)) - D_1



colors = sns.color_palette("bright")
fig = plt.figure(2, figsize=(5, 4))
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

ax.set(xlabel=r'$D_1$', ylabel=r'$D_2$')
ax.minorticks_on()
ax.grid(alpha=0.5)
ax.grid(alpha=0.3, which='minor')

ax.plot(D_1,(D_2), color=colors[0])


handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc="lower right", ncol=1, bbox_to_anchor=(1, 0.23))
fig.tight_layout(rect=[0, 0.0, 1, 0.99])
# plt.savefig("MD_dist_bound_exa_2.pdf")
plt.show()

