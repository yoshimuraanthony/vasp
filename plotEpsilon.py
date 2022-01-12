#!/usr/bin/env python3

from numpy import array
import matplotlib.pyplot as plt

def plotIm(
        ebounds = [0, 5],
        infile = 'imag.dat',
        infile2 = 'imag_pris.dat',
        label = 'FeMoS$_2$',
        label2 = 'MoS$_2$',
        save = True,
        outfile = 'imag.png',
        legend = True,
        ):
    """
    plots imaginary componentent of frequency-dependent dielectric tensor
    infile: text file containing imaginary components (str)
        * can be generated from plot.sh
            (cms.mpi.univie.ac.at/wiki/index.php/Dielectric_properties_of_SiC)
    outfile: file to which plot is written (str)
    """
    en_list = []
    imag_list = []
    with open(infile) as f:
        for line in f:
            en, imag = (float(val) for val in line.split())
            en_list.append(en)
            imag_list.append(imag)

    en2_list = []
    imag2_list = []
    with open(infile2) as f:
        for line in f:
            en, imag = (float(val) for val in line.split())
            en2_list.append(en)
            imag2_list.append(imag)

    dif_ar = array(imag_list) - array(imag2_list)

    fig, ax = plt.subplots()
    ax.set_xlabel('energy (eV)', fontsize = 14)
    ax.set_ylabel(r'Im($\epsilon$) (a.u.)', fontsize = 14)
    ax.set_xlim(ebounds[0], ebounds[1])

    ax.plot(en_list, imag_list, label = label)
    ax.plot(en2_list, imag2_list, label = label2)
    ax.plot(en_list, dif_ar, label = 'difference')
    ax.axhline(y = 0, color = 'k')
    ax.grid()
    if legend:
        ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(outfile)
    plt.show()    

#---------------------------------- Old ---------------------------------------
# Freqency-Dependent Dielectric Tensor
# Anthony Yoshimura
# 11/17/16

# Requires vasprun.xml

import numpy as np
import matplotlib.pyplot as plt
import re

def getImaginary(vasprun = 'vasprun.xml'):
    """
    plots imaginary component of frequency-dependent dielectric tensor
    vasprun.xml: vasprun file (str)
    """
    # import imaginary component
    vr_str = open(vasprun).read()
    im_str = re.findall(r'imag(.*?)/imag', vr_str, re.DOTALL)
    im_list = im_str[0].split('\n')
    im_ar = [row.split() for row in im_list]

    # sift out all non numbers
    im_tab = []  
    for strings in im_ar[11:-3]:
        row = [float(val) for val in strings if (val != '<r>' and val != '</r>')]
        im_tab.append(row)

    # list of directional compoenents with respect to efield for plotting
    im_plot_tab = np.transpose(im_tab) 

    print(im_tab[:10])

    # plot
    plt.plot(im_plot_tab[0], im_plot_tab[1], label = 'X')
    plt.plot(im_plot_tab[0], im_plot_tab[2], label = 'Y')
    plt.title('Frequency-Dependent Dielectric Tensor', fontsize = 20)
    plt.xlabel('Energy (eV)', fontsize = 16)
    plt.ylabel(r'$\epsilon^2$ (a.u.)', fontsize = 16)
    plt.xlim(0,6)
    plt.grid()
    plt.legend(loc='best')

    plt.savefig('ep.pdf')
    plt.show()

