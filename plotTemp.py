# Anthony Yoshimura
# 09/26/2017

from numpy import array
import matplotlib.pyplot as plt
import os
import re

def plot(infile = 'OSZICAR', outfile = 'tempComp.pdf', tlim = 900, save = False, show = True):
    """
    plots temperature fluctuation during MD simulations
    ** note: OSZICAR files should be named OSZICAR_0.1
    infile: prefix shared by all temperature data files (str)
    """
    files = [f for f in os.listdir('.') if infile in f and '.pdf' not in f]
    pref_len = len(infile)

    temp_tab = []
    label_list = []
    for fi in files:
        with open(fi) as f:
            temp_list = []
            for line in f:
                if 'T' in line:
                    temp = float(line.split()[2])
                    temp_list.append(temp)
            temp_tab.append(temp_list)
            label = fi[pref_len:]
            label_list.append(label)

    # obtain time steps from OSIZCAR suffixes, i.e., OSZICAR_1.0fs --> 10
    timestep_list = [float(re.sub("\D", "", label)) for label in label_list]
    time_tab = [[n * timestep_list[m] / 10 for n in range(len(temp_tab[m]))]\
               for m in range(len(label_list))]

    # determine zorder of plots
#    tup_list = [(max(temp_tab[n]), n) for n in range(3)]
#    tup_list.sort()
#    max_dict = {}
#    for n in range(3):
#        max_dict[n] = 2000 - int(max(temp_tab[n]))

    fig, ax = plt.subplots()
    ax.set_title('Temperature fluctuation', fontsize = 14)
    ax.set_xlabel('time (fs)', fontsize = 14)
    ax.set_ylabel('temperature (K)', fontsize = 14)
#    ax.set_xlim(0, tlim)
    ax.set_xlim(0, max(time_tab[0]))
    ax.grid()
    for n in range(len(label_list)):
        start = int(tlim * 10 / timestep_list[n])
#        plt.plot(time_tab[n][:start], temp_tab[n][-start:], label = label_list[n], zorder = max_dict[n])
        plt.plot(time_tab[n][:start], temp_tab[n][-start:], label = label_list[n])

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(outfile)
    if show:
        plt.show()

