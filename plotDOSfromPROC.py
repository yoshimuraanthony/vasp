from numpy import pi, log, arange, cos, sin, sqrt, linspace, zeros, exp
from numpy import floor, ceil
from numpy import array, transpose
from copy import deepcopy
from periodic import table as ptable
import matplotlib.pyplot as plt

"""
Goal: plot projected density of states from PROCAR file
    * assumes LORBIT = 10 for l and site projections
"""

l_dict = {'s': 1, 'p': 2, 'd': 3, 'f': 4, 'tot': -1}

def plot(
        # dos parameters
        ebounds = [-3, 5],
        proj_list = ['s', 'p', 'd'],
        nedos = 2000,
        sigma = 0.06,
        maxDen = 'auto',
        subtractFermi = True,
        getTotal = True,

        # input files
        OUTCAR = 'OUTCAR',
        PROCAR = 'PROCAR',

        # plot text
        title = 'l-projected DOS',
        xlabel = 'energy (eV)',
        ylabel = 'density of states (a.u.)',
        grid = False,

        # saving
        figsize = None,
        save = False,
        outfile = 'orbDOS.png',
        transparent = False,
        ):
    """
    plots densty of states
    """
    fig, ax = plt.subplots(figsize = figsize)

    if subtractFermi:
        efermi = getEFermi(OUTCAR)
        energy_ar, dos_dict = getPDOS(array(ebounds) + efermi, proj_list, nedos,
                                      sigma, OUTCAR, PROCAR, getTotal)
        energy_ar -= efermi
        xlabel = 'E - E$_f$ (eV)'

    else:
        energy_ar, dos_dict = getPDOS(array(ebounds), proj_list, nedos, sigma,
                                      OUTCAR, PROCAR, getTotal)

    for proj in proj_list:
        dos_ar = dos_dict[proj]
        ax.plot(energy_ar, dos_ar, label = proj)

    # text
    ax.legend()
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel(ylabel, fontsize = 14)
    ax.get_yaxis().set_ticks([]) # units are arbitrary
    ax.text(.02, .98, title, va='top', ha='left', transform=ax.transAxes,
            fontsize=14)

    # plot boundaries
    ax.set_xlim(ebounds[0], ebounds[1])
    if maxDen == 'auto':
        maxDen = max(dos_ar) * 1.2
    ax.set_ylim(0, maxDen)

    if grid:
        ax.grid()

    # saving
    plt.tight_layout()
    if save:
        plt.savefig(outfile, transparent = transparent)

    plt.show()


def getPDOS(ebounds = [-10, 10],
        proj_list = ['s','p','d'],
        nedos = 2000,
        sigma = 0.2,
        OUTCAR = 'OUTCAR',
        PROCAR = 'PROCAR',
        getTotal = True,
        ):
    """
    returns list of lists of energies and (gaussian smeared) densities
    ebounds: energy domain over which dos is taken (list of 2 floats)
    nedos: number of points considered in energy domain (pos int)
    sig: standard deviation of gaussian functions (float)
    proj_list: list of projections
        site: pos ints
        orbitals: 's', 'p', 'd', 'f'
    OUTCAR: OUTCAR file from vasp run (str)
    PROCAR: PROCAR file generated from LORBIT = 10 run (str)
    getTotal: if True, totel density is included in weight_dict (bool)
    """
    eigenval_list = getEigenvals(OUTCAR)

    # use full energy range if bounds are not specified
    if type(ebounds) != list and type(ebounds) != tuple:
        ebounds = (min(eigenval_list), max(eigenval_list))

    # energy domain
    energy_ar = linspace(ebounds[0], ebounds[1], nedos)

    if type(proj_list[0]) == int:
        weight_dict = getSiteWeights(proj_list, PROCAR, getTotal)
    elif type(proj_list[0]) == str:
        weight_dict = getLWeights(proj_list, PROCAR, getTotal)
    else:
        weight_dict = getSiteAndLWeights(proj_list, PROCAR, getTotal)

    dos_dict = {}
    for proj in weight_dict:
        weight_list = weight_dict[proj]
        dos_ar = zeros(nedos)

        for eigenval, weight in zip(eigenval_list, weight_list):
            dos_ar += weight * gaussian(energy_ar, eigenval, sigma)

        dos_dict[proj] = dos_ar

    return energy_ar, dos_dict
    

def getSiteWeights(site_list = [1], PROCAR = 'PROCAR', getTotal = True):
    """
    returns list of eigenvalues and their corresponding weights for a
        specified projection
    proj_list: list of atomic sites onto which DOS is projected
        (list of pos ints)
    PROCAR: PROCAR file generated from LORBIT = 10 run (str) 
    getTotal: if True, totel density is included in weight_dict (bool)
    """
    # UNDER CONSTRUCTION: does not support spin orbit coupling calculations
    if getTotal:
        site_list.append('tot')

    weight_dict = {}
    for site in site_list:
        weight_dict[site] = []

    with open(PROCAR) as f:
        for line in f:
            for site in site_list:
                if str(site) in line[:3]:
                    weight_dict[site].append(float(line.split()[4]))

    return weight_dict


def getLWeights(l_list = ['s'], PROCAR = 'PROCAR', getTotal = True):
    """
    returns list of eigenvalues and their corresponding weights for a
        specified projection
    proj_list: list of orbital l's onto which DOS is projected
        (list of orbital symbols, 's', 'p', 'd', 'f')
    PROCAR: PROCAR file generated from LORBIT = 10 run (str)
    getTotal: if True, totel density is included in weight_dict (bool)
    """
    # UNDER CONSTRUCTION: does not support spin orbit coupling calculations
    if getTotal:
        l_list.append('tot')

    weight_dict = {}
    for l in l_list:
        weight_dict[l] = []

    with open(PROCAR) as f:
        for line in f:
            if 'tot' in line[:3]:
                for l in l_list:
                    weight_dict[l].append(float(line.split()[l_dict[l]]))

    return weight_dict


def getSiteAndLWeights(pair_list = [1], PROCAR = 'PROCAR', getTotal = True):
    """
    returns list of eigenvalues and their corresponding weights for a
        specified projection
    pair_list: list of pairs of sites and l values onto which DOS is projected
        (list of tuples, e.g. (75, 'd'))
    PROCAR: PROCAR file generated from LORBIT = 10 run (str)
    getTotal: if True, totel density is included in weight_dict (bool)
    """
    # UNDER CONSTRUCTION: does not support spin orbit coupling calculations
    if getTotal:
        pair_list.append(('tot', 'tot'))

    weight_dict = {}
    for pair in pair_list:
        weight_dict[pair] = []

    with open(PROCAR) as f:
        for line in f:
            for pair in pair_list:
                site, l = pair
                if str(site) in line[:3]:
                    weight_dict[pair].append(float(line.split()[l_dict[l]]))

    return weight_dict


def getEFermi(OUTCAR = 'OUTCAR'):
    """
    returns fermi energy
    OUTCAR: OUTCAR file from VASP run (str)
    """
    eigenval_list = []
    with open(OUTCAR) as f:
        for line in f:
            if 'E-fermi' in line:
                efermi = float(line.split()[2])
                break

    return efermi


def getEigenvals(OUTCAR = 'OUTCAR'):
    """
    returns list of eigenvalues
    OUTCAR: OUTCAR file from VASP run (str)
    """
    eigenval_list = []
    with open(OUTCAR) as f:
        for line in f:
            if 'NBANDS' in line:
                nbands = int(line.split()[-1])

            if 'band energies' in line:
                for band in range(nbands):
                    eigenval_list.append(float(f.readline().split()[1]))

    return eigenval_list

#----------------------------- Smearing functions -----------------------------

def gaussian(x, mu, sigma):
    """
    returns value predicted by gaussian curve
    x: input value (float)
    mu: mean (float)
    sig: standard deviation (pos float)
    """
    return 1. / (sqrt(2.*pi)*sigma)*exp(-((x - mu)/sigma)**2/2)

#---------------------------------- Scratch -----------------------------------

def oldGetEigenvalsAndWeights(proj = 1, PROCAR = 'PROCAR'):
    """
    returns lists of eigenvalues and their corresponding weights for a
        specified projection
    proj: orbital or atomic site onto which band structure is projected
        site: pos int
        orbital: e.g. 's', 'py', 'dxy', etc.
    PROCAR: PROCAR file generated from LORBIT = 10 run (str)
    """
    # UNDER CONSTRUCTION: does not support spin orbit coupling calculations
    eigenval_list = []
    weight_list = []

    with open(PROCAR) as f:
        f.readline()

        # get number of k-points, bands, and ions
        nkpts, nbands, nions = [int(val) for val in f.readline().split()[3::4]]
        print('nkpts = %s, nbands = %s, nions = %s' %(nkpts, nbands, nions))
        f.readline()

        # get eigenvalues and site-projections for each k-point and band
        if type(proj) == int:

            for k in range(nkpts):
                f.readline()
                f.readline()

                for j in range(nbands):
                    eigenval_list.append(float(f.readline().split()[4]))

                    for i in range(proj + 1):
                        f.readline()

                    weight_list.append(float(f.readline().split()[4]))

        # get eigenvalues and l-projections for each k-point and band
        else:

            for k in range(nkpts):
                f.readline()
                f.readline()

                for j in range(nbands):
                    eigenval_list.append(float(f.readline().split()[4]))

                    for i in range(nions + 2):
                        f.readline()

                    weight_list.append(float(f.readline().split()[4]))


    return eigenval_list



#        if subtractFermi:
#            efermi = getFermi(DOSCAR)
#            energy_ar, dos_ar = getDOS(array(ebounds) + efermi, nedos, sigma,
#                                       EIGENVAL)
#            ax.plot(energy_ar - efermi, dos_ar, label=label1, color=color1)
#            xlabel = 'E - E$_f$ (eV)'
#
