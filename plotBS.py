# Band structure plots
# Anthony Yoshimura
# 10/24/16

# Current directory should contain EIGENVAL, POSCAR, DOSCAR, and special.kpt

from numpy import array, zeros, cross, dot, pi, transpose, sort, append, arange, ceil
from numpy.linalg import norm
from getChemForm import getChemForm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os


def getFermiEnergy(DOSCAR = 'DOSCAR', printInfo = True):
    """
    Returns Fermi energy
    DOSCAR: DOSCAR file (str)
    """
    try:
        with open(DOSCAR, 'r') as f:
            efermi = float(f.readlines()[5].split()[3])

    except FileNotFoundError:
        print("Could not open DOSCAR file at %s.\n\tFermi energy set to zero"
              %DOSCAR)
        efermi = 0.0

    if printInfo:
        print('Fermi level at: ', efermi)
    return efermi


def getRecipLattice(POSCAR = 'POSCAR'):
    """
    Returns 3x3 array whose rows are reciprocal lattice vectors
    POSCAR: POSCAR file (str)
    """
    # get real-space lattice from POSCAR
    try: 
        with open(POSCAR, 'r') as f:
            real_lattice = array(
                [list(map(float, f.split())) for f in f.readlines()[2:5]])

    except FileNotFoundError:
        print("Could not find file POSCAR at %s.\n\t\
              will assume lattice is simple cubic!" %POSCAR)
        real_lattice = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])

    # 3x3 place holder for reciprocal lattice, calculated in the next loop
    lattice = zeros([3, 3])

    # take the cross product of each pair of real-space vectors
    for i in range(3):
        j = (i + 1) % 3 
        k = (i + 2) % 3 
        crossProduct = cross(real_lattice[j], real_lattice[k])
        volume = dot(real_lattice[i], crossProduct)
        lattice[i] = 2 * pi * crossProduct / volume

    return lattice


def getEigenVals(
        EIGENVAL = 'EIGENVAL',
        POSCAR = 'POSCAR',
        DOSCAR = 'DOSCAR',
        spinPolarized = False,
        subtractFermi = True,
        start_kpt = [0, 0, 0],
        printInfo = True,
        ):
    """
    Returns dictionary with the following {key: value} pairs

        'band_tab'  :  list of lists of eigenvalues along kpath
        'band2_tab' :  list of lists of eigenvalues for spin down
        'nkpts'     :  number of kpoints along kpath
        'nbands'    :  number of eigenvalues per kpoint
        'kpt_list'  :  distances of kpoints along kpath
        'nelect'    :  number of electrons
        'efermi'    :  fermi energy

    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    subtractFermi: if True, Fermi energy is subtracted from
        all eigenvalues (bool)
    printInfo: if True, prints info from EIGENVAL and DOSCAR files
        while running (bool)
    """
    # UNDER CONSTRUCTION: find symmetry points from discontinuities in path
    #                     start from first k-point in KPOINTS or OUTCAR
    # get Fermi level from DOSCAR
    efermi = getFermiEnergy(DOSCAR, printInfo)
    if subtractFermi:
        e0 = efermi
    else:
        e0 = 0.0

    # get reciprocal lattice
    lattice = getRecipLattice(POSCAR)

    # obtain eigenvalues along k-path from EIGENVAL file
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
        eigenval2_tab = [] # lists of eigenvalues at every kpoint
        kpt_list = [] # distances of kpoints along kpath
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
            kpt_list.append(distance)
    
            # record eigenvalues at current kpoint
            eigenval_list = []

            # no spin polarization. one eigenval per k-point
            if not spinPolarized:

                for j in range(nbands):

                    try:
                        line = f.readline()
                        eigenval = float(line.split()[1])

                    except IndexError:
                        print("coudn't find eigenvalue for k-point %s and\
                                band %s" %(i, j))
                        print(line)
                        break

                    eigenval_list.append(eigenval - e0)  # e0 is fermi energy
                                                         # or zero

                eigenval_tab.append(eigenval_list)

            # spin polarization. two eigenvals per k-point
            if spinPolarized:
                eigenval2_list = []
            
                for j in range(nbands):

                    try:
                        line = f.readline()
                        eigenval, eigenval2 = [float(val) for val in
                                               line.split()[1:]]

                    except IndexError:
                        print("coudn't find spin-down eigval for k-point %s\
                              and band %s" %(i, j))
                        print(line)
                        break

                    eigenval_list.append(eigenval - e0)  # e0 is fermi energy
                                                         # or zero
                    eigenval2_list.append(eigenval2 - e0)

                eigenval_tab.append(eigenval_list)
                eigenval2_tab.append(eigenval2_list)
        
    # transpose table into list of bands to plot in pyplot
    band_tab = transpose(eigenval_tab)
    band2_tab = transpose(eigenval2_tab)

    # add end of path to break list
    break_list.append(nkpts - 1)

    return {
        'band_tab':   band_tab,
        'band2_tab':  band2_tab,
        'nkpts':      nkpts,
        'nbands':     nbands,
        'kpt_list':   kpt_list,
        'nelect':     nelect,
        'efermi':     efermi,
        'break_list': break_list
        }

def getFoldedEigenVals(subtractFermi = True):
    """
    returns dictionary for folded bandstructure of supercell
    dim: dimenstion of super cell (list of 3 pos ints)
    substractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    """
    # UNDER CONSTRUCTION: allow for spin-polarized plots
    # get data from directories corresponding to each run
    dir_list = next(os.walk('.'))[1]
    dir_list.sort()

    foldedBand_tab = []
    efermi_list = []
    nelect = 0
    nbands = 0
    for d in dir_list:
        print('Getting values from %s' %d)
        eigen_dict = getEigenVals(
                        EIGENVAL = '%s/EIGENVAL' %d,
                        POSCAR = '%s/POSCAR' %d,
                        DOSCAR = '%s/DOSCAR' %d,
                        subtractFermi = False)

        foldedBand_tab.append(eigen_dict['band_tab'])
        efermi_list.append(eigen_dict['efermi'])
        nelect += eigen_dict['nelect']
        nbands += eigen_dict['nbands']

    foldedBand_tab = array(foldedBand_tab).reshape(nbands, -1)
    sort(foldedBand_tab)
    efermi = max(efermi_list)
    if subtractFermi:
        foldedBand_tab -= efermi

    return {'band_tab':   foldedBand_tab,
            'nkpts':      eigen_dict['nkpts'],
            'nbands':     nbands,
            'kpt_list':   eigen_dict['kpt_list'],
            'nelect':     nelect,
            'efermi':     efermi,
            'break_list': eigen_dict['break_list']}


def getBandGap(
        EIGENVAL = 'EIGENVAL',
        POSCAR = 'POSCAR',
        DOSCAR = 'DOSCAR',
        so = False,
        subtractFermi = False,
        printInfo = True,
        ):
    """
    Returns dictionary containing bandgap parameters
    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    so: True if calculation included spin-orbit coupling (bool)
    subtractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    """
    if printInfo:
        print('calculating band gap')
    eigen_dict = getEigenVals(EIGENVAL, POSCAR, DOSCAR, subtractFermi = subtractFermi, printInfo = printInfo)

    # get energies
    band_tab = eigen_dict['band_tab']
    if subtractFermi:
        efermi = 0
    else:
        efermi = eigen_dict['efermi']
    
    # list of band maxima and minima in increasing order
    max_list = array([band.max() for band in band_tab])
    min_list = array([band.min() for band in band_tab])

    vb_dif = cb_dif = max_list.max() - min_list.min()

    # find band whose max is closest to fermi energy
    for val in max_list:
        new_dif = abs(efermi - val)
        if new_dif < vb_dif:
            vb_dif = new_dif
            vb_num = max_list.tolist().index(val)

    # bands within five of closest band
    vb_try = [vb_num + n for n in range(-5, 6)]

    # check for gap
    biggestGap = 0
    for n in vb_try:
        vb, cb = n, n + 1
        vbm = max_list[vb]
        cbm = min_list[cb]
        bandgap = cbm - vbm
        # choose biggest gap
        if bandgap > .01 and bandgap > biggestGap:
            biggestGap = bandgap
            vb_index = band_tab[vb].tolist().index(vbm)       # check if direct
            cb_index = band_tab[cb].tolist().index(cbm)
            trueVb, trueCb = vb, cb
            trueVbm, trueCbm = vbm, cbm
            if printInfo:
                if vb_index == cb_index:
                    print('band gap: %s (direct at k-point %s)' %(bandgap, vb_index))
                else:
                    print('band gap: %s (indirect)' %bandgap)
                    print('vbm at k-point %s, and cbm at k-point %s' %(vb_index, cb_index))

    if biggestGap > .01:
        return {'gap': biggestGap, 'cb': trueCb, 'vb': trueVb, 'efermi': efermi,
                'cbm': trueCbm, 'vbm': trueVbm, 'k_cbm': cb_index, 'k_vbm': vb_index}
            
    else:
        print('material has no band gap')


#--------------------------------- PLOTTING -----------------------------------
def plot(# energy range and reference
         bounds        = [-5, 4],
         shift         = 0,
         subtractFermi = False,
         etickSpacing  = 1,  # eV

         # input files
         EIGENVAL      = 'EIGENVAL',
         EIGENVAL2     = None,
         DOSCAR        = 'DOSCAR',
         DOSCAR2       = None,
         POSCAR        = 'POSCAR',

         # k-path
         spec_kpts     = 'special.kpt',
         start_kpt     = [0, 0, 0],

         # appearance
         color         = 'blue',
         color2        = 'red',
         transparent   = False,

         # additional plotting objects
         highlights    = [],
         spinPolarized = False,
         showFermi     = False,
         showFermi2    = False,
         forSlides     = False,
         label_list    = ['spin up', 'spin down'],
         legend        = False,
         showGap       = False,
         title         = 'auto',

         # saving
         figsize       = [6,6],
         save          = False,
         outfile       = 'bs.pdf',
         ):
    """
    Produces band structure plot bandstructure.pdf

    bounds: lower and upper energy bounds in eV (tuple of 2 floats)
        * if None, plots all eigenvalues
    EIGENVAL: EIGENVAL file (str)
    EIGENVAL2: EIGENVAL file to be compared to EIGENVAL (str of None)
    DOSCAR: DOSCAR file (str)
    DOSCAR: DOSCAR file corresponding to EIGENVAL2 (str)
    POSCAR: POSCAR file (str)
    spec_kpts: list of or file containing high-symmetry kpoints on kpath (list of str)
    highlights: list of bands to be highlighted (list of ints)
    figsize: figure's width by height in inches (tuple of 2 floats)
    subtractFermi: if True, fermi-energy is subtracted from eigenvalues (bool)
    showFermi: if True, plots dashed line at fermi energy (bool)
    showGap: if True, Egap is displayed with arrow connecting vbm and cbm (bool)
    title: title of plot (str)
        * if 'auto', title = '%s band structure' %(chemical formula)
    forSlides: if True, axis and text are white (bool)
    """
    # UNDER CONSTRUCTION: read speciel kpt labels and distances from KPOINTS files
    dir_list = next(os.walk('.'))[1]
#    dir_list = [1]

    # folded band structure from multiple EIGENVAL files
    if len(dir_list) > 1 and ('f' in dir_list[0] or 'F' in dir_list[0]):
        print('folded plot')
        eigen_dict = getFoldedEigenVals(subtractFermi = subtractFermi)

    # stitch k-path segments together into one k-path
#    elif len(dir_list) > 1 and ('s' in dir_list[0] or 'S' in dir_list[0]):
#        eigen_dict = getStitchedEigenVals(subtractFermi = subtractFermi)

    # regular band structure
    else:
        print('getting single plot')
        eigen_dict = getEigenVals(EIGENVAL, POSCAR, DOSCAR, spinPolarized, subtractFermi, start_kpt)

    band_tab    = eigen_dict['band_tab'] + shift
    nkpts       = eigen_dict['nkpts']
    kpt_list    = eigen_dict['kpt_list']
    efermi      = eigen_dict['efermi']
    break_list  = eigen_dict['break_list']
    print('nkpts = %s\nbreak_list = %s' %(nkpts, break_list))

    # plot spin down component
    if spinPolarized:
        band2_tab = eigen_dict['band2_tab']

    # compare second band structures
    if type(EIGENVAL2) == str:
        eigen_dict2 = getEigenVals(EIGENVAL2, POSCAR, DOSCAR2, subtractFermi, start_kpt)
        band2_tab   = eigen_dict2['band_tab']
        efermi2     = eigen_dict2['efermi']

    # create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    font = {'fontname':'Times'}

    # plot bands
    if type(highlights) == int:
        highlights = [highlights]

    for band in range(len(band_tab)):
        if band + 1 in highlights:
            ax.plot(kpt_list, band_tab[band], color = 'red', linewidth = 3, zorder = -1)
        else:
            ax.plot(kpt_list, band_tab[band], color = color, linewidth = 2, zorder = -4)

    if type(EIGENVAL2) == str or spinPolarized:
        for band in range(len(band2_tab)):
            ax.plot(kpt_list, band2_tab[band], color = color2, linewidth = 2, zorder = -2, label = 'SO')

    # x- and y-bounds
    if bounds != None:
        ax.set_ylim(bounds)
    ax.set_xlim(0, kpt_list[-1])

    # energy tick spacing
    if type(etickSpacing) == float or type(etickSpacing) == int:
        yticks = arange(ceil(bounds[0]), bounds[1], etickSpacing)
        ax.set_yticks(yticks)

    # title and labels
    if title == 'auto':
        chemForm = getChemForm(POSCAR)
        title = '%s band structure' %chemForm
    if title != False:
        ax.set_title(title, fontsize = 14)
    ax.set_xlabel('k-points', fontsize = 14)

    # Fermi level
    if subtractFermi:
#        ax.set_ylabel('$\mathrm(\mathsf{E - E_f}$ (eV)', fontsize = 14)
        ax.set_ylabel('$E - E_f$ (eV)', fontsize = 14)
        ax.axhline(y = 0, linestyle = 'dashed')
    else:
        ax.set_ylabel('Energy (eV)', fontsize = 14)
        if showFermi and not showFermi2:
#            ax.axhline(y = efermi, linestyle = 'dashed')
            ax.axhline(y = efermi, linestyle = 'dashed', linewidth = 2, zorder = 4, color = 'gray')
        if showFermi2:
            ax.axhline(y = efermi, linestyle = 'dashed', color = 'red')
            ax.axhline(y = efermi2, linestyle = 'dashed', color = 'black')

    # show band gap with arrow
    if showGap:
        gap_dict = getBandGap(EIGENVAL = EIGENVAL, DOSCAR = DOSCAR, POSCAR = POSCAR, subtractFermi = subtractFermi)
        gap = gap_dict['gap']
        vbm = gap_dict['vbm']
        cbm = gap_dict['cbm']
        k_vbm = int(gap_dict['k_vbm'])
        k_cbm = int(gap_dict['k_cbm'])
        if k_vbm == k_cbm:
            gap_type = 'direct'
        else:
            gap_type = 'indirect'

        kdist_vbm = kpt_list[k_vbm]
        kdist_cbm = kpt_list[k_cbm]
        dk = kdist_cbm - kdist_vbm
        height = (bounds[1] - bounds[0]) / 30
        width = kpt_list[-1] / 40
        #print('cbm: %.3g, vbm: %.3g, k_cbm: %s, k_vbm: %s, k_dist_cbm: %.3g, kdist_vbm: %.3g'
        #      %(cbm, vbm, k_cbm, k_vbm, kdist_cbm, kdist_vbm))
        ax.annotate('', xy = (kdist_cbm, cbm), xytext = (kdist_vbm, vbm),
                    xycoords = 'data', textcoords = 'data',
                    arrowprops = dict(facecolor = 'red', edgecolor = 'red',
                    shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8))

        ypos = (cbm + vbm) / 2
        valign = 'center'
        if (kdist_vbm + kdist_cbm) / 2 > len(kpt_list):
            xpos = kpt_list[-1] * .03
            halign = 'left'
        else:
            xpos = kpt_list[-1] * .97
            halign = 'right'

        ax.text(xpos, ypos, '%s gap\n%.4g eV' %(gap_type, gap), fontsize = 14,
                 verticalalignment = valign, horizontalalignment = halign)
    
    # special kpoint ticks
    if type(spec_kpts) == str:
        spec_kpts = [sym.strip('\n') for sym in open(spec_kpts).readlines() if len(sym) > 1]

    nsub = len(spec_kpts) - 1
#    tickPositions = [kpt_list[int((nkpts - 1) * n / nsub)] for n in range(nsub + 1)]
    tickPositions = [kpt_list[b] for b in break_list]
    ax.set_xticks(tickPositions)
    ax.set_xticklabels(spec_kpts)
    symLineColor = 'black'

    # axis and tick color
    if forSlides:
        for spine in ax.spines:
            ax.spines[spine].set_color('white')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis = 'x', colors = 'white', labelsize = 14) 
        ax.tick_params(axis = 'y', colors = 'white', labelsize = 14) 

        symLineColor = 'white'
        transparent = True

    # vertical lines above ticks
#    for n in range(0, nkpts, int(nkpts / (len(spec_kpts) - 1))):
    for n in break_list:
        ax.axvline(x = kpt_list[n], color = symLineColor, linewidth = 2)
    print('break_list = %s' %break_list)

    # thicker axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)

    # legend for comparing BS
    if legend:
        custom_lines = [Line2D([0], [0], color=color, lw=2),
                        Line2D([0], [0], color=color2, lw=2)]
        ax.legend(custom_lines, label_list)

    plt.tight_layout()
    if save:
        plt.savefig(outfile, transparent = transparent)

    plt.show()

#------------------------------------------------------------------------------

# calling from terminal
if __name__ == '__main__':
   plot(subtractFermi = True)

#-------------------------------- SCRATCH -------------------------------------

def getStitchedEigenVals(subtractFermi = True):
    """
    returns dictionary for folded bandstructure of supercell
    dim: dimenstion of super cell (list of 3 pos ints)
    substractFermi: if True, Fermi energy is subtracted from all eigenvalues (bool)
    """
    dir_list = next(os.walk('.'))[1]
    dir_list.sort()
    stitchedBand_tab = []
    efermi_list = []
    kpt_list = []
    totalDist = 0
    nkpts = 0
    for d in dir_list:
        print('Getting values from %s' %d)
        eigen_dict = getEigenVals(
                        EIGENVAL = '%s/EIGENVAL' %d,
                        POSCAR = '%s/POSCAR' %d,
                        DOSCAR = '%s/DOSCAR' %d,
                        subtractFermi = subtractFermi)

        newKpt_list = [dist + totalDist for dist in eigen_dict['kpt_list']]
        kpt_list += newKpt_list
        nbands = eigen_dict['nbands']
        if len(stitchedBand_tab) == 0:
            stitchedBand_tab = [[] for n in range(nbands)]
        for band in range(nbands):
            stitchedBand_tab[band] += (eigen_dict['band_tab'][band]).tolist()
        efermi_list.append(eigen_dict['efermi'])
        nkpts += eigen_dict['nkpts']
        totalDist = kpt_list[-1]

    efermi = max(efermi_list)
    return {'band_tab':  array(stitchedBand_tab),
            'nkpts':     nkpts,
            'nbands':    eigen_dict['nbands'],
            'kpt_list':  kpt_list,
            'nelect':    eigen_dict['nelect'],
            'efermi':    efermi}

#    except FileNotFoundError:
#        print("Could not find %s -- aborting" %EIGENVAL)
#        sys.exit()
