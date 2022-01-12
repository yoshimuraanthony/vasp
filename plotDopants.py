from numpy import array, zeros, cross, dot, pi, transpose, sort, append, arange, ceil
from numpy.linalg import norm
from getChemForm import getChemForm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

def getBars(
        OUTCAR = 'OUTCAR'
        ):
    """
    Returns dictionary with the following {key: value} pairs
        * assumes ISPIN = 2

        'band_tab'   :  list of lists of eigenvalues along kpath
        'nelect'     :  number of electrons
        'nkpts'      :  number of kpoints along kpath
        'nbands'     :  number of eigenvalues per kpoint
        'kDist_list' :  distance travesed along kpath at each k-point
        'nelect'     :  number of electrons
        'efermi'     :  fermi energy

    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    subtractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    printInfo: if True, prints info from EIGENVAL and DOSCAR files while running (bool)
    """
    # obtain nbands, nelect, and eigenvalues from OUTCAR
    with open(OUTCAR, 'r') as f:
        for line in f:
            if 'NBANDS' in line:
                nbands = int(line.split()[-1])
            if 'NELECT' in line:
                nelect = int(line.split()[2])
            if 'spin component 1' in line:
                for n in range(3): # get to energies and occupations
                    f.readline()
      

        
    # obtain eigenvalues along k-path from EIGENVAL
    try:
        with open(EIGENVAL, 'r') as f:

            for i in range(5):  # useful info starts at line 6
                f.readline()

            # number of electrons, kpoints, and bands
            line = f.readline()
    
            nelect, nkpts, nbands = [int(val) for val in line.split()]
            if printInfo:
                print('number of electrons:', nelect)
                print('number of k-points:', nkpts)
                print('nunber of bands:', nbands)
    
            # place holders
            eigenval_tab = [] # lists of eigenvalues at every kpoint
            kDist_list = [] # distances of kpoints along kpath
            break_list = [] # path-indices at which the path turns or breaks
            cart_kpt = zeros(3) # Gamma point position in cartesian recirocal space
            distance = 0 # total distance travelled along kpath

            # get eigenvalues and kpath distances
            for i in range(nkpts):  # skips line before data
                f.readline()

                # kpoint position in reciprocal lattice coordinates
                kpt = array([float(val) for val in f.readline().split()[:3]])

                # kpoint position in cartesian
                old_cart_kpt = cart_kpt
                cart_kpt = dot(kpt, lattice)

                # cartesian shift between adjacent kpoints
                shift = norm(cart_kpt - old_cart_kpt)

                if shift > 0.3:  # if path is discontinuous, start anew
                    shift = 0

                if shift < 1E-10:  # record where path breaks or changes direction
                    break_list.append(i)

                # record total cartesian distance travelled along kpath
                distance += shift
                kDist_list.append(distance)

                # record eigenvalues at current kpoint
                eigenval_tab.append([])
                for j in range(nbands):
                    try:
                        eigenval = float(f.readline().split()[1])
                    except IndexError:
                        print("coudn't find eigenvalue for k-point %s and band %s" %(i, j))
                    eigenval_tab[-1].append(eigenval - e0)  # e0 is fermi energy or zero

    except FileNotFoundError:
        print("Could not find %s -- aborting" %EIGENVAL)
        sys.exit()

    # transpose table into list of bands to plot in pyplot
    band_tab = transpose(eigenval_tab)

    # add end of path to break list
    break_list.append(nkpts - 1)

    return {'band_tab':   band_tab,
            'nkpts':      nkpts,
            'nbands':     nbands,
            'kDist_list':   kDist_list,
            'nelect':     nelect,
            'efermi':     efermi,
            'break_list': break_list}


#-------------------------- Lists and Dictionaries -------------------------------

root = '/Users/anthonyyoshimura/Desktop/koratkar/MoS2/dopants/'
dop_list = ['Nb', 'Tc', 'Ta', 'Re']
system_list = ['iso', 'line', 'cluster3s', 'cluster3h']
#dop_list = ['Nb']
#color_list = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
color_list = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
size_dict = { 
    'iso'      : array([3, 4, 7, 9, 12, 13, 16]),
    'cluster3s': array([9, 12, 21, 27, 36, 48]),
    'cluster3h': array([9, 12, 21, 27, 36, 48]),
    'line'      : array([3, 4, 5, 6, 7, 8, 9]),
    }   
title_dict = { 
    'iso'      : 'isolated dopant',
    'cluster3s': 'cluster centered on chalcogen',
    'cluster3h': 'cluster centered on hole',
    'line'     : 'line of dopants',
    }   


