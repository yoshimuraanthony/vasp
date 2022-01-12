from numpy import pi, log, arange, cos, sin, sqrt, linspace, zeros, exp
from numpy import floor, ceil
from numpy import array, transpose
from periodic import table as ptable
import matplotlib.pyplot as plt


def plot(
        # dos settings
        ebounds = [-3, 3],
        nedos = 2000,
        sigma = 0.06,
        maxDen = 'auto',
        subtractFermi = True,

        # input files
        EIGENVAL = 'EIGENVAL',
        EIGENVAL2 = None,
        DOSCAR = 'DOSCAR',
        OUTCAR = 'OUTCAR',
        DOSCAR2 = None,
        OUTCAR2 = None,

        # plot text
        title = None,
        xlabel = 'energy (eV)',
        ylabel = 'D.O.S. (a.u.)',
        label = 'GW',
        label2 = 'LDA',

        # aesthetics
        color = (0, 0, 1),
        color2 = (1, .5, 0),
        legend = False,

        # saving
        figsize = (6, 5),
        save = False,
        outfile = 'dos.png',
        transparent = False,
        dpi = 300
        ):
        """
        plots densty of states
            * efermi is subtracted
        """
        fig, ax = plt.subplots(figsize = figsize)

        nelect, nkpt, nband = getNelectNkptsNbands(OUTCAR)
        print('nelect = %s, nkpt = %s, nband = %s' %(nelect, nkpt, nband))

        if subtractFermi:
            efermi = getFermi(DOSCAR)
        else:
            efermi = 0

        energy_ar, dos_ar = getDOS(array(ebounds) + efermi, nedos, sigma,
                                   EIGENVAL)
        dos_ar /= nelect * nkpt

        # subtracting Fermi level
        ax.plot(energy_ar - efermi, dos_ar, label=label, color=color,
                zorder=3)

        # semi-transparent color fill below DOS
        fill_ar = array([energy - efermi for energy in energy_ar
            if energy <= efermi])
#        ax.fill_between(fill_ar, 0, dos_ar[:len(fill_ar)], color=color,
#                alpha=0.5, zorder=3)

        # plot second dos
        if type(EIGENVAL2) == str:

            nelect2, nkpt2, nband2 = getNelectNkptsNbands(OUTCAR2)
            print('nelect2 = %s, nkpt = %s, nband = %s' %(nelect2, nkpt2,
                nband2))
            if subtractFermi:
                efermi2 = getFermi(DOSCAR2)
            else:
                efermi2 = 0

            ebounds_shifted = array(ebounds) + efermi2
            energy2_ar, dos2_ar = getDOS(ebounds_shifted, nedos, sigma,
                                         EIGENVAL2)
            
#            dos2_ar /= nelect2 * nkpt2 * 1.5
#            dos2_ar /= nelect2 * nkpt2 
            dos2_ar /= nelect2 * nkpt2 / 2
            ax.plot(energy2_ar - efermi2, dos2_ar, label = label2,
                    color=color2)

            fill2_ar = array([energy2 - efermi2 for energy2 in energy2_ar
                if energy2 <= efermi2])
#            ax.fill_between(fill2_ar, 0, dos2_ar[:len(fill2_ar)], color=color2,
#                    alpha=0.8)

        # text
        ax.set_xlabel(xlabel, fontsize = 14)
        ax.set_ylabel(ylabel, fontsize = 14)
        ax.get_yaxis().set_ticks([]) # units are arbitrary
        ax.text(.99, .98, title, va='top', ha='right', transform=ax.transAxes,
                fontsize=14)
        ax.tick_params(axis = 'x', labelsize = 14) 
        ax.tick_params(axis = 'y', labelsize = 14) 

        # plot boundaries
        ax.set_xlim(ebounds[0], ebounds[1])

        if maxDen == 'auto':
            maxDen = max(dos_ar) * 1.3
        ax.set_ylim(0, maxDen)

        if legend:
            ax.legend(loc = 1)
        # saving
        plt.tight_layout()
        if save:
            plt.savefig(outfile, dpi = dpi, transparent = transparent)

        plt.show()


def getDOS(ebounds = [-10, 10],
        nedos = 2000,
        sigma = 0.2,
        EIGENVAL = 'EIGENVAL'):
    """
    returns list of lists of energies and (gaussian smeared) densities)
    ebounds: energy domain over which dos is taken (list of 2 floats)
    nedos: number of points considered in energy domain (pos int)
    sig: standard deviation of gaussian functions (float)
    EIGENVAL: EIGENVAL file from vasp run (str)
    """
    eigenval_list = getEigenvals(EIGENVAL)
#    if type(ebounds) != list and type(ebounds) != tuple:
#        ebounds = (min(eigenval_list), max(eigenval_list))
    energy_ar = linspace(ebounds[0], ebounds[1], nedos)

    dos_ar = zeros(nedos)
    for eigenval in eigenval_list:
        dos_ar += gaussian(energy_ar, eigenval, sigma)

    return energy_ar, dos_ar
    

def getEigenvals(EIGENVAL = 'EIGENVAL'):
    """
    returns lists of eigenvalues
    EIGENVAL: EIGENVAL file from VASP run (str)
    """
    eigenval_list = []

    with open(EIGENVAL) as f:
        if 'EIG' in EIGENVAL:
            for n in range(5):
                f.readline()
    
            nelect, nkpts, nbands = [int(val) for val in f.readline().split()]
    
            for i in range(nkpts):
                f.readline()
                f.readline()
    
                for j in range(nbands):
#                    eigenval_list.append(float(f.readline().split()[1]))
                    eigenval_list += [float(val) for val in
                            f.readline().split()[1:]]

#    print('number of eigenvalues = %s' %len(eigenval_list))


        elif 'OUT' in EIGENVAL:
            print('Grabbing eigenvalues from DOSCAR')
            for line in f:
                if 'NKPTS' in line:
                    nkpts = int(line.split()[3])
                    nbands = int(line.split()[-1])

                if 'NELECT' in line:
                    nelect = int(float(line.split()[2]))

                if 'spin component 1' in line:

                    for i in range(nkpts):
                        for n in range(4):
                            f.readline()

                        for j in range(nbands):
                            eigenval_list.append(
                                float(f.readline().split()[2]))
    
    return eigenval_list
            

#def getFermi(DOSCAR = 'OUTCAR'):
def getFermi(DOSCAR = 'DOSCAR'):
    """
    returns Fermi energy
        * Note: GW OUTCARs don't have Fermi energy
    OUTCAR: OUTCAR file from VASP run (str)
    """
    with open(DOSCAR) as f:
        for n in range(5):
            f.readline()
        efermi = float(f.readline().split()[3])
#        for line in f:
#            if 'E-fermi' in line:
#                efermi = float(line.split()[2])

    return efermi


def getNelectNkptsNbands(OUTCAR = 'OUTCAR'):
    """
    returns number of electrons
    OUTCAR: OUTCAR file from VASP run (str)
    """
    with open(OUTCAR) as f:
        for line in f:
            if 'NELECT' in line:
                nelect = float(line.split()[2])
            elif 'NKPTS' in line:
                nkpts = float(line.split()[3])
                nbands = float(line.split()[-1])
        

    return nelect, nkpts, nbands
    

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
def oldGetFermi(OUTCAR = 'OUTCAR'):
    """
    returns fermi energy
    OUTCAR: OUTCAR file from VASP run (str)
    """
    with open(OUTCAR) as f:
        for n in range(5):
            for line in f:
                if 'E-fermi' in line:
                    efermi = float(line.split()[2])
                    break

    return efermi

#            max1 = max(dos_ar)
#                max2 = max(dos2_ar)
#                ratio = max1/max2
