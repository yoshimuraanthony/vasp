# Site-Projected Band Structure
# Anthony Yoshimura
# 10/26/16

# Current directory should contain PROCAR, POSCAR, DOSCAR, and special.kpt
# Please see HOWTOplotPBS for instructions

from numpy import array, zeros, cross, dot, pi, transpose, sort, append, arange, ceil 
from numpy.linalg import norm 
from getChemForm import getChemForm 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D 
import sys 
import os
from plotBS import getFermiEnergy, getRecipLattice


def getEigenVals(
        proj = 1,
        PROCAR = 'PROCAR',
        POSCAR = 'POSCAR',
        DOSCAR = 'DOSCAR',
        subtractFermi = True,
        so = False,
        ):
    """
    Returns dictionary with the following {key: value} pairs

        'band_tab'    :  list of lists of energies along k-path
        'band_wt_tab' :  site_contribution at every kpoint along each band
        'nkpts'       :  numper of kpoints along k-path
        'nbands'      :  number of eigenvalues per kpoint
        'kpt_list'    :  distances of kpoints along k-path
        'nions'       :  number of ions

    proj: orbital or atomic site onto which band structure is projected
        for site: pos int
        for orbital: e.g. 's', 'py', 'dxy', etc.
    PROCAR: PROCAR file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    subtractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    so: set True for spin-orbit calculations (bool)
    """
    # option to subtract Fermi energy taken from DOSCAR
    if subtractFermi:
        efermi = getFermiEnergy(DOSCAR)
    else:
        efermi = 0.0

    # get reciprocal lattice from POSCAR
    lattice = getRecipLattice(POSCAR)

    # obtain eigenvalues from PROCAR
    try:
        with open(PROCAR, 'r') as f:

            f.readline()    # useful info starts at line 2

            # get number of k-points, bands, and ions
            nkpts, nbands, nions = [int(val) for val in f.readline().split()[3::4]]
            print('nkpts = %s, nbands = %s, nions = %s' %(nkpts, nbands, nions))
        
            # PROCAR format depends on whether or not SOC is included
            if so:
                so_skip = 3*(nions + 1)    # skip lines to ignore magnetization data
            else:
                so_skip = 0
        
            f.readline()    # kpoint data starts at line 4
        
            kpt_wt_tab = []     # site/orb-contribution of every band at each k-point
            eigval_tab = []     # eigenvalues at every kpoint for each site/orb
            kpt_list = []       # distances of k-points along k-path
            break_list = []     # path-indices at which the path turns or breaks
            cart_kpt = zeros(3) # Gamma point position in cartesian recirocal space
            distance = 0        # total distance travelled along k-path
        
            # get eigenvalues and k-path distances
            for k in range(nkpts):

                # k-point position in reciprocal lattice coordinates
                kpt = array([float(val) for val in f.readline().split()[3:6]])
        
                # k-point position in cartesian coordinates
                old_cart_kpt = cart_kpt
                cart_kpt = dot(kpt, lattice)
        
               # cartesian shift between adjacent kpoints
                shift = norm(cart_kpt - old_cart_kpt)

                if shift > 0.3:  # if path is discontinuous, start anew
                    shift = 0

                if shift < 1E-10:  # record where path breaks or changes direction
                    break_list.append(k)

                # record total cartesian distance travelled along k-path
                distance += shift
                kpt_list.append(distance)
        
                f.readline()           # move to eigenvalue data
        
                # record eigenvalues at current k-point
                eigval_tab.append([])
                kpt_wt_tab.append([])

                # get info from current k-point
                for j in range(nbands):

                    # get eigenvalues from each band
                    eigenval = float(f.readline().split()[4])
                    eigval_tab[-1].append(eigenval - efermi)
        
                    # get site projection
                    if type(proj) == int:

                        # get total weight from site
                        for i in range(proj + 1):
                            f.readline()   # move to site contribution
            
                        wt = float(f.readline().split()[4])
                        kpt_wt_tab[-1].append(wt)
    
                        # move to next eigenvalue
                        for i in range(nions + so_skip - proj + 2):
                            f.readline()

                    # get orbital projection
                    else:

                        # get total weight from orbital
                        for i in range(nions + 2):
                            f.readline()   # move to site contribution
            
#                        col = header_list.index(proj)
                        col = header_dict[proj]
                        wt = float(f.readline().split()[col])
                        # print('wt:',wt)
                        kpt_wt_tab[-1].append(wt)
    
                        # move to next eigenvalue
                        for i in range(so_skip + 1):
                            f.readline()
        
                # move to next kpoint 
                f.readline()

    # quit if PROCAR is not read
    except IOError:
        print("Could not open %s -- aborting" %PROCAR)
        sys.exit()

    # always label last k-point                   
    break_list.append(k)

    # list of bands plotable in pyplot
    band_tab = transpose(eigval_tab)
    
    # site-contribution at every kpoint along each band    
    band_wt_tab = transpose(kpt_wt_tab) * 50

    return {'band_tab':    band_tab,
            'band_wt_tab': band_wt_tab,
            'nkpts':       nkpts,
            'nbands':      nbands,
            'kpt_list':    kpt_list,
            'nions':       nions,
            'break_list':  break_list,
            'efermi':      efermi,
            }


#----------------------------------- PLOTTING ----------------------------------------
def plot(proj = 1,

        # input files
        PROCAR = 'PROCAR',
        PROCAR2 = None,
        DOSCAR = 'DOSCAR',
        POSCAR = 'POSCAR',
        spec_kpts = 'special.kpt',

        # plot properties
        title = 'auto',
        bounds = 'auto',
        weight = 1,
        subtractFermi = True,
        so = False,
        background = True,
        figsize = (5, 5),

        # saving
        save = False,
        outfile = 'auto',
        dpi = 300,
        ):
    """
    Produces site-projected band structure plot pbs_{site}.pdf

    proj: orbital or atomic site onto which band structure is projected
        for site: pos int
        for orbital: e.g. 's', 'py', 'dxy', etc.
    bounds: lower and upper energy bounds in eV (list of floats)
    PROCAR: PROCAR file (str)
    PROCAR2: PROCAR (same NBANDS and NPTS) file to be compared to PROCAR
        (str or None)
    DOSCAR: DOSCAR file (str)
    POSCAR: POSCAR file (str)
    spec_kpts: high-symmetry kpoints on k-path (list of str)
    wieght: scale determining size of points (float)
    subtractFermi: if True, Fermi energy is subtracted from all eigenvalues
        (bool)
    so: set True for spin-orbit calculations (bool)
    background: if True, full band structure is plotted in the background
        (bool)
    """
    # extract all relavent values
    eigen_dict = getEigenVals(proj, PROCAR, POSCAR, DOSCAR, subtractFermi, so)

    band_tab    = eigen_dict['band_tab']
    band_wt_tab = eigen_dict['band_wt_tab']
    nkpts       = eigen_dict['nkpts']
    kpt_list    = eigen_dict['kpt_list']
    break_list  = eigen_dict['break_list']
    efermi      = eigen_dict['efermi']

    # extract second set of values for comparison
    if type(PROCAR2) == str:
        eigen_dict2 = getEigenVals(site, PROCAR2, POSCAR, DOSCAR,
                subtractFermi, so)

        band_tab2    = eigen_dict2['band_tab']
        band_wt_tab2 = eigen_dict2['band_wt_tab']

    # create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    # font = {'fontname':'Times'}

    # plot bands
    for band in range(len(band_tab)):
        ax.scatter(kpt_list, band_tab[band], s = band_wt_tab[band] * weight,
                                            color = 'blue', zorder = -1)

        if type(PROCAR2) == str:
            ax.scatter(kpt_list, band_tab2[band], s = band_wt_tab2[band],
                                            color = 'gray', zorder = -2)

        if background:
            ax.plot(kpt_list, band_tab[band], color = 'silver',
                                        linewidth = 1, zorder = -3)

#    ax.set_ylim(bounds[0], bounds[1])
    if bounds == 'auto':
        if subtractFermi:
            bounds = (-3, 3)
        else:
            bounds = (efermi - 3, efermi + 3)
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_xlim(0, kpt_list[-1])

    # title and labels
    if title == 'auto':
        if type(proj) == int:
            title = 'Projected band structure for site %s' %proj
        else:
            title = 'Projected band structure for %s' %proj
    if title != None:
        ax.set_title(title, fontsize = 18)

    ax.set_xlabel('K-point', fontsize = 16)

    if subtractFermi:
        ax.set_ylabel('$E - E_f$ (eV)', fontsize = 16)
        ax.axhline(y = 0, color = 'gray', ls = '--')

    else:
        ax.set_ylabel('Energy (eV)', fontsize = 16)
    # ax.set_ylabel('$\mathrm{Energy}$ $\mathrm{(eV)}$')
    # ax.set_xlabel('$\mathrm{Kpoints}$')
    
    # special kpoint ticks
    if type(spec_kpts) == str:
        with open(spec_kpts) as f:
            spec_kpts = [sym.strip('\n') for sym in f.readlines() if len(sym) > 1]

    nsub = len(spec_kpts) - 1
    tickPositions = [kpt_list[b] for b in break_list]
    ax.set_xticks(tickPositions)
    ax.set_xticklabels(spec_kpts)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)

    # vertical lines about ticks
    for n in break_list:
        ax.axvline(x = kpt_list[n], color = 'black', linewidth = 2)

    # thicker axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)

    # output and save
    plt.tight_layout()
    if save:
        if outfile == 'auto':
            if type(PROCAR2) == str:
                plt.savefig('pbs_%s_comp.png' %proj, format = 'png', dpi = dpi)
            else:
                plt.savefig('pbs_%s.png' %proj, format = 'png', dpi = dpi)
        else:
            plt.savefig(outfile)
    plt.show()


def plotCV(site,
        cbounds = [0.75, 1.75],
        vbounds = [-1.25, -0.25],
        PROCAR        = 'PROCAR',        PROCAR2       = None,
        DOSCAR        = 'DOSCAR',        POSCAR        = 'POSCAR',
        spec_kpts     = 'special.kpt',   subtractFermi = False,
        so            = False,           background    = False):
    """
    Produces site-projected band structure plot pbs_cv_{site}.pdf

    site: atomic site to be projected (positive int)
    cbounds: lower and upper energy bounds for cb in eV (list of floats)
    vbounds: lower and upper energy bounds for vb in eV (list of floats)
    PROCAR: PROCAR file (str)
    PROCAR2: PROCAR (same NBANDS and NKPTS) file to be compared to PROCAR (str or None)
    DOSCAR: DOSCAR file (str)
    POSCAR: POSCAR file (str)
    spec_kpts: high-symmetry kpoints on k-path (list of str)
    subtractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    so: set True for spin-orbit calculations (bool)
    background: if True, full band structure is plotted in the background (bool)
    """
    eigen_dict = getEigenVals(site, PROCAR, POSCAR, DOSCAR, subtractFermi, so)

    band_tab    = eigen_dict['band_tab']
    band_wt_tab = eigen_dict['band_wt_tab']
    nkpts       = eigen_dict['nkpts']
    kpt_list    = eigen_dict['kpt_list']

    if type(PROCAR2) == str:
        eigen_dict2 = getEigenVals(site, PROCAR2, POSCAR, DOSCAR, subtractFermi, so)

        band_tab2    = eigen_dict2['band_tab']
        band_wt_tab2 = eigen_dict2['band_wt_tab']

    # create figure with two axes sharing x-axes
    fig, (ax1, ax2) = plt.subplots(2, sharex = True, figsize = (6, 8))
    fig.subplots_adjust(hspace = 0.05)

    font = {'fontname':'Times'}

    # plot bands
    for band in range(len(band_tab)):
        ax1.scatter(kpt_list, band_tab[band], s = band_wt_tab[band],
                                            color = 'red', zorder = -1)
        ax2.scatter(kpt_list, band_tab[band], s = band_wt_tab[band],
                                            color = 'red', zorder = -1)

        if type(PROCAR2) == str:
            ax1.scatter(kpt_list, band_tab2[band], s = band_wt_tab2[band],
                                            color = 'gray', zorder = -2)
            ax2.scatter(kpt_list, band_tab2[band], s = band_wt_tab2[band],
                                            color = 'gray', zorder = -2)

        if background:
            ax1.plot(kpt_list, band_tab[band], color = 'silver',
                                        linewidth = 2, zorder = -3)
            ax2.plot(kpt_list, band_tab[band], color = 'silver',
                                        linewidth = 2, zorder = -3)

    ax1.set_ylim(cbounds)
    ax2.set_ylim(vbounds)
    ax1.set_xlim(0, kpt_list[-1])
    ax2.set_xlim(0, kpt_list[-1])
#    ax.set_ylabel('$\mathrm{Energy}$ $\mathrm{(eV)}$')
#    ax.set_xlabel('$\mathrm{Kpoints}$')

    # title and labels
    ax1.set_title('Projected band structure for site ' + str(site), fontsize = 18)
    ax2.set_xlabel('K-point', fontsize = 14)

    if subtractFermi:
        fig.text(0.02, 0.55, '$E - E_f$ (eV)', rotation = 'vertical', fontsize = 14)
    else:
        fig.text(0.02, 0.55, 'Energy (eV)', rotation = 'vertical', fontsize = 14)
    
    # special kpoint ticks
    ax1.tick_params(axis = 'x', which = 'both', length = 0) # remove x-axis tick lines
    ax2.tick_params(axis = 'x', which = 'both', length = 0)

    if type(spec_kpts) == str:
        spec_kpts = [sym.strip('\n') for sym in open(spec_kpts).readlines() if len(sym) > 1]

    nsub = len(spec_kpts) - 1
    tick_positions = [kpt_list[int((nkpts - 1) * n / nsub)] for n in range(nsub + 1)]

    ax2.set_xticks(tick_positions) # high-symmetry point symbols
    ax2.set_xticklabels(spec_kpts, fontsize = 14)

    for n in range(0, nkpts, int(nkpts / (len(spec_kpts) - 1))):
        ax1.axvline(x = kpt_list[n], color = 'black', linewidth = 2) # vertical lines
        ax2.axvline(x = kpt_list[n], color = 'black', linewidth = 2)

    # thicker axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(3)
        ax2.spines[axis].set_linewidth(3)

    # merge plots
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # broken axes
    d = .02  # length of diagonal line indicating broken axis
    kwargs = dict(transform = ax1.transAxes, color = 'k', clip_on = False, linewidth = 3)
    ax1.plot((-d, d), (-d - .01, d - .01), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d - .01, d - .01), **kwargs)

    kwargs.update(transform = ax2.transAxes, linewidth = 3)
    ax2.plot((-d, +d), (1.01 - d, 1.01 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1.01 - d, 1.01 + d), **kwargs)

    if type(PROCAR2) == str:
        plt.savefig('pbs_cv_' + str(site) + '_comp.pdf', format = 'pdf')
    else:
        plt.savefig('pbs_cv_' + str(site) + '.pdf', format = 'pdf')
    plt.show()

#----------------------------- HELPERS -----------------------------

# list of labels in PROCAR projections
#header_list = ['ion', 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2', 'tot']
#header_list = ['ion', 's', 'p', 'd', 'tot']
header_dict = {'ion': 0, 's': 1, 'p': 2, 'd': 3,
               'px': 2, 'py': 3, 'pz': 4, 'dxy': 5,
               'dyz': 6, 'dz2': 7, 'dxz': 8, 'dx2': 9}

#header_list = ['ion', 's', 'p', 'd', 'tot']

#------------------------------ NOTES ------------------------------
# plotting the fat band structure for a specified site
# 
# The current directory should contain the following output files from
# the VASP band structure calculation
#   -PROCAR
#   -DOSCAR
#   -POSCAR
# 
# An additional file, special.kpt, can be created specifying the
# high-symmetry kpoints along the k-path.  The file should contain
# a single column of symbols.  For example
# 
# $\Gamma$
# $M$
# $T$
# $\Gamma$
# 
# where the $ signs instruct pyplot to use mathmode text.
# 
# With these files in place, start the python interpreter and import
# plotPBS.  Then run plotPBS.plot() for a specfied site.  For example,
# the fourth site can be plotted with
# 
# >>> plotPBS.plot(4)
# 
# This plot would be saved in the current directory as 'pbs_4.pdf'
# 
# A second band structure can be plotted simultaneously by entering the
# optional argument PROCAR2, e.g.
# 
# >>> plotBS.plot(4, POSCAR2 = '../otherdirectory/PROCAR')
# 
# To see all options, execute
# 
# >>> help(plotPBS.plot)
# 
# Lastly, the plot function can also be called from the command line using
# 
# $ python -c "import plotPBS; plotPBS.plot(4)"
#
#
# NOTE: the script requires numpy and matplotlib modules

# NOTE: if output writes: WARNING: dimensions on CHGCAR file are different
#        set INCAR tags NG{X,Y,Z}F = CHGCAR grid dimensions

# NOTE: SO includes magnetization projections.  We ignore those here

#---------------------------- SCRATCH ---------------------------------
#def getFermiEnergy(DOSCAR = 'DOSCAR'):
#    """
#    Returns Fermi energy
#    DOSCAR: DOSCAR file (str)
#    """
#    try:
#        inputFile_DOSCAR = open(DOSCAR)          
#
#    except IOError:
#        print("Could not open DOSCAR file, will define Fermi enery as zero")
#        efermi=0.0
#
#    else:
#        efermi = float(inputFile_DOSCAR.readlines()[5].split()[3])
#        inputFile_DOSCAR.close()
#
#    print('Fermi level at: ', efermi, 'eV')
#    return efermi
#
#
#def getRecipLattice(POSCAR = 'POSCAR'):
#    """
#    Returns 3x3 array whose rows are reciprocal lattice vectors
#    POSCAR: POSCAR file (str)
#    """
#    try: 
#        inputFile_POSCAR = open(POSCAR)
#
#    except IOError:
#        print("Could not find file POSCAR; will assume simple cubic!")
#        lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#
#    else:
#        real_lattice = np.array(
#            [list(map(float, f.split())) for f in inputFile_POSCAR.readlines()[2:5]])
#        inputFile_POSCAR.close()
#
#        lattice = np.zeros([3, 3])
#
#        for i in range(3):
#            j = (i + 1) % 3 
#            k = (i + 2) % 3 
#            cross = np.cross(real_lattice[j], real_lattice[k])
#            volume = np.dot(real_lattice[i], cross)
#            lattice[i] = 2 * np.pi * cross / volume
#
#        return lattice
#
