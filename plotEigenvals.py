# Anthony Yoshimura
# 09/03/18

from numpy import arange, array, zeros, cross, dot, pi, transpose, sort, append
from numpy.linalg import norm
from getChemForm import getChemForm
import matplotlib.pyplot as plt 
import os
from plotBS import getFermiEnergy, getRecipLattice, getBandGap
from plotEnCon import getPaths

blue_list = [(1, .5, 0), (.2, .2, 1), (.2, .2, .8), (.2, .2, .6), (.2, .2, .4)] 

def getDirectories(targFiles = ['EIGENVAL', 'POSCAR', 'DOSCAR'], top = '.'):
    """
    returns a list of directories that contain desired files
    file_list: list of files that directories must contain (list of str)
    """
    if type(targFiles) == str:
        targFiles = [targFiles]

    dir_list = []
    for root, dirs, files in os.walk(top):
        soFarSoGood = True
        for targFile in targFiles:
            if targFile not in files: 
                soFarSoGood = False
                break
        if soFarSoGood:
            dir_list.append(root)

    return dir_list


def getEigenvalTab(
        EIGENVAL = 'EIGENVAL',
        DOSCAR = 'DOSCAR',
        kpoint = [1/3, 1/3, 0],
        printInfo = False,
        atomsPerCell = 3,
        walk = True,
        ):
    """
    returns table containing number of atoms and eigenvalues at specified k-point 
    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    kpoint: k-point at which eigenvalues are read (list of 3 floats)
    printInfo: if True, prints band gaps and Fermi energies as they are read (bool)
    """
    # get paths for EIGENVAL, POSCAR, and DOSCARs
    if walk:
        EIGENVAL_list = getPaths(EIGENVAL)
        DOSCAR_list = getPaths(DOSCAR)
    else:
        EIGENVAL_list = [EIGENVAL]
        DOSCAR_list = [DOSCAR]

    # get eigenvalues belonging to desired k-point from EIGENVALs
    eigenval_tab = []
    for EIGENVAL, DOSCAR in zip(EIGENVAL_list, DOSCAR_list):

        # get Fermi energies from DOSCAR
        with open(DOSCAR) as f:
            efermi = getFermiEnergy(DOSCAR, printInfo = False)

        # get eigenvalues from EIGENVAL
        if printInfo:
            print('getting eigenvalues from %s' %EIGENVAL)
        with open(EIGENVAL) as f:

            # get number of atoms and bands
            natoms = int(f.readline().split()[0])
            if natoms < 4:
                natoms = 0 # zero Ga atoms in pristine cell

            for n in range(4):
                f.readline()
            nelect, nkpts, nbands = [int(val) for val in f.readline().split()]
            if printInfo:
                print('nelect = %s, natoms = %s, nbands = %s' %(nelect, natoms, nbands))

            # if more than k-point was used, find point closest to the desired k-point
            if nkpts > 1:
                for line in f:
                    line_list = line.split()
                    if len(line_list) == 4:
                        trialKpoint = [float(val) for val in line_list[:-1]]
    
                        # check if it's the correct k-point
                        match = True 
                        for desiredComp, trialComp in zip(kpoint, trialKpoint):
                            dif = abs(desiredComp - trialComp)
                            if abs(desiredComp - trialComp) > 0.001:
                                match = False
                                break
                   
                        # break from loop when k-point is found
                        if match:
                            break

            # if only one k-point was used
            else:
                f.readline()
                line_list = f.readline().split()[:-1]
                trialKpoint = [float(val) for val in line_list[:-1]]

            if printInfo:
                print('taking eigenvalues at %s' %trialKpoint)
    
            # get eigenvalues
            eigenval_list = [float(f.readline().split()[1]) for n in range(nbands)]
                
        # add to eigenvalue table
        eigenval_tab.append((int(natoms / atomsPerCell), eigenval_list, efermi))

    return eigenval_tab
                

def getBars(
        ebounds = [-4, 0.75],
        kpoint = [1/3, 1/3, 0],
        printInfo = False,
        thres = 0.3628, #eV (default for Ga-doping project)
        getPris = True,
        prisRoot = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands',
        ):
    """
    returns dictionary of energy ranges for bars representing vbm, cbm, and midgap states
    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    kpoint: k-point at which eigenvalues are read (list of 3 floats)
    printInfo: if True, prints band gaps and Fermi energies as they are read (bool)
    thres: threshold for gap in eV (float)
    getPris: if True, gets energy ranges from pristine system
    root: directory containing VASP output files for pristine system
    """
    eigenval_tab = getEigenvalTab('EIGENVAL', 'DOSCAR', kpoint, printInfo)
    emin, emax = ebounds
    emax += 1.5 # make sure conduction band goes above top of plot

    # get eigenvalues from pristine system
    if getPris:
        EIGENVAL = '%s/%s' %(prisRoot, 'EIGENVAL')
        DOSCAR = '%s/%s' %(prisRoot, 'DOSCAR')
        eigenval_tab += getEigenvalTab(EIGENVAL, DOSCAR, kpoint, printInfo, walk = False)
        
    # get all eigenvalues below top directory
    bar_dict = {}
    for natoms, eigenval_list, efermi in eigenval_tab:

        # use larger thres for pristine system
        if natoms == 0:
            thres = 1.6

        # find of energy ranges in which eigenvalues are closely spaced
        top_list = []
        bot_list = [emin]
        for n in range(len(eigenval_list) - 1):
            thisEig = eigenval_list[n]
            nextEig = eigenval_list[n + 1]

            # only look at eigenvalues within ebounds
            if thisEig > emin and thisEig < emax:
                if nextEig - thisEig > thres:
                    top_list.append(thisEig)
                    bot_list.append(nextEig)
        
        # restore original threshold value
        if natoms == 0:
            thres -= 1.5

        top_list.append(emax)
   
        bar_dict[natoms] = top_list, bot_list, efermi

    return bar_dict


def getEdges(
        ebounds = [-4, 0.75],
        thres = 0.1356, #eV (default for Ga-doping project)
        printInfo = False,
        getPris = True,
        prisRoot = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands',
        ):
    """
    returns dictionary containing band edges in eV within deired energy range
    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    thres: threshold for gap in eV (float)
    getPris: if True, gets energy ranges from pristine system
    root: directory containing VASP output files for pristine system
    """
    # get list of directories containing required files
    root_list = getDirectories(targFiles = ['EIGENVAL', 'DOSCAR'], top = '.')

    # get pristine band edges
    if getPris:
        root_list.append(prisRoot)

    # dictionary containing edge values
    edge_dict = {}

    # loop through root directories
    for root in root_list:
        DOSCAR = '%s/DOSCAR' %root
        EIGENVAL = '%s/EIGENVAL' %root

        # get Fermi level from DOSCAR
        efermi = getFermiEnergy(DOSCAR, printInfo = False)
    
        # obtain eigenvalues along k-path from EIGENVAL
        if printInfo:
            print('getting eigenvalues from %s' %EIGENVAL)

        with open(EIGENVAL, 'r') as f:
    
            # get number or atoms and bands
            natoms = int(int(f.readline().split()[0]) / 3)
            if natoms < 4:
                natoms = 0
            print('natoms = %s' %natoms)

            for i in range(4):  # useful info starts at line 6
                f.readline()
    
            # number of electrons, kpoints, and bands
            nelect, nkpts, nbands = [int(val) for val in f.readline().split()]
            if printInfo:
                print('nelect = %s, natoms = %s, nbands = %s' %(nelect, natoms, nbands))
    
            # place holders
            eigenval_tab = [] # lists of eigenvalues at every kpoint
    
            # get eigenvalues and kpath distances
            for i in range(nkpts):
                for n in range(2): # skips k-points lines before eigenvals
                    f.readline()
    
                # record eigenvalues at current kpoint
                eigenval_list = []
                for j in range(nbands):
                    eigenval = float(f.readline().split()[1])
                    eigenval_list.append(eigenval)
    
                eigenval_tab.append(eigenval_list)
    
        # transpose table into list of bands to plot in pyplot
        band_tab = transpose(eigenval_tab)
    
        # find band maxima and minima
        max_list = array([band.max() for band in band_tab])
        min_list = array([band.min() for band in band_tab])
        
        # find all bands in ebounds
        lowerBound, upperBound = ebounds
    
        # find lowest band in ebounds
        for index, en in enumerate(min_list):
            if en > lowerBound:
                lowestIndex = index
                break
            
        # find highest band in ebounds
        for index, en in enumerate(reversed(max_list)):
            if en < upperBound:
                highestIndex = nbands - index
                break
        if highestIndex == nbands:
            highestIndex -= 1 # ensure that search for cbm stays in bounds of min_list
    
        # find highest cbm and lowest vbm
        if printInfo:
            print('searching for gap in bands %s through %s' %(lowestIndex, highestIndex))
        top_list = [] # top of band (bottom of gap)
        bot_list = [] # bottom of band (top of gap)
        for index in range(lowestIndex, highestIndex):
            vbm = max_list[index]
            cbm = min_list[index + 1]
            bandgap = cbm - vbm
            if bandgap > thres:
                if printInfo:
                    print('found gap of size %s eV above band %s' %(bandgap, index))
                top_list.append(vbm)
                bot_list.append(cbm)
    
        vbm, cbm = min(top_list), max(bot_list)
        print('vbm = %s, cbm = %s' %(vbm, cbm))
        edge_dict[natoms] = vbm, cbm

    return edge_dict


def getGaps(
        bar_dict,
        edge_dict,
        ):
    """
    returns dictionary containing band gaps in eV within deired energy range
    bar_dict: output of getBars (dict)
    edge_dict: output of getEdges (dict)
    """
    gap_dict = {}

    for natoms in bar_dict:
        top_list, bot_list, efermi = bar_dict[natoms]
        vbm, cbm = edge_dict[natoms]

        # get indirect gap
        indGap = cbm - vbm

        # find top closest to vbm
        minDist = indGap
        for top in top_list:
            dif = abs(vbm - top)
            if dif < minDist:
                minDist = dif
                dVbm = top

        # find bottom closest to cbm
        minDist = indGap
        for bot in bot_list:
            dif = abs(cbm - bot)
            if dif < minDist:
                minDist = dif
                dCbm = bot
        
        # get direct gap
        dGap = dCbm - dVbm

        gap_dict[natoms] = dGap, indGap

    return gap_dict


def plotBars(
        ebounds = [-4, 0.75],
        kpoint = [1/3, 1/3, 0],
        printInfo = False,
        showFermi = True,
        showGaps = True,
        barThres = 0.3628, # eV (default for Ga-doping project)
        edgeThres = 0.1356,
        color_list = 'auto',
        width = 0.9,
        figsize = [6,5],
        save = False,
        outfile = 'bars.pdf',
        ):
    """
    plots eigenvalues for various concentrations
    kpoint: k-point at which eigenvalues are read (list of 3 floats)
    printInfo: if True, prints band gaps and Fermi energies as they are read (bool)
    showFermi: if True, fermi levels are plotted as black dashed lines (bool)
    showGaps: if True, gap values are shown underneath the cbm's (bool)
    barThres: energy threshold above which an unoccupied region is considered a gap (float)
    edgeThres: energy threshold above which an unoccupied region is considered a gap (float)
    """
    # UNDER CONSTRUCTION:
    #     * length of faded bar extensions and positions of gap values
    #       depend on edgeThres, which shouldn't be the case!
    #     * gaps and concentrations should round with correct number of
    #       sig figs.  Right now it is tailored to work for G a-doping
    bar_dict = getBars(ebounds, kpoint, printInfo, barThres)
    edge_dict = getEdges(ebounds, edgeThres, printInfo) # small thres to count all gaps
    gap_dict = getGaps(bar_dict, edge_dict)
    numBars = len(bar_dict)

    # sort number of atoms corresponding to ascending Ga concentration
    natoms_list = [natoms for natoms in bar_dict]
    natoms_list.sort(reverse = True)
    natoms_list.insert(0, natoms_list.pop(-1))

    # label bars with Ga concentration %
    label_list = []
    for natoms in natoms_list:
        if natoms == 0:
            label_list.append('0.00')
        else:
            label = str(100 / natoms)
            label += '000'
            label_list.append('%.4s' %label)
            
    # prepare figure
    fig, ax = plt.subplots()
    
    # plot bars and edges labelled by ascending Ga concentration
    for natoms, n in zip(natoms_list, range(numBars)):

        print('natoms = %s' %natoms)

        # plot bars
        top_list, bot_list, efermi = bar_dict[natoms]

        # same color for each concentration
        if color_list == 'auto':
            barColor = next(ax._get_lines.prop_cycler)['color']
        else:
            barColor = color_list[n]
    
        # plot bars
        for top, bot in zip(top_list, bot_list):
            print('bot = %s, top = %s' %(bot, top))
            ax.bar(left = n, height = top - bot, width = width + .05, bottom = bot,
                   color = barColor)

        # plot edges if they are significantly different from a bar's edge
        vbm, cbm = edge_dict[natoms]
        endPointsX = [n - width/2, n + width/2]
        barEdges_list = top_list + bot_list

        # check if indirect gap is significatnly smaller than bars' gap
        tracker = 0 # distinguish vbm from cbm
        for bandEdge in [vbm, cbm]:
            tracker += 1
            significant = True
            for barEdge in barEdges_list:
                dif = abs(bandEdge - barEdge)
                if dif < 0.01:
                    significant = False
                    break

            # plot faded extension to bar showing true band edge
            if significant:
                if tracker % 2 == 0:
                    print('plotting a cbm extension')
                    bot, top = bandEdge, bandEdge + 0.5

                else:
                    print('plotting a vbm extension')
                    bot, top = bandEdge - 0.5, bandEdge

                ax.bar(left = n, height = top - bot, width = width + .05, bottom = bot,
                       color = barColor, zorder = -1, alpha = .4)

        # plot fermi levels
        if showFermi:
            endPointsY = [efermi, efermi]
            ax.plot(endPointsX, endPointsY, color = 'red', linewidth = 2, linestyle = 'dashed')

        # show band gap values
        if showGaps:
            dGap, indGap = gap_dict[natoms]
            xpos = n
            ypos = cbm - .05
            text = '%.3g eV' %dGap

            if abs(dGap - indGap) > 0.05:
                text = '%.3g (%.4s)' %(dGap, indGap)
            ax.text(xpos, ypos, text, va = 'top', ha = 'center', fontsize = 11)

    # figure properties
    ax.set_ylim(ebounds)
#    ax.set_title('Energy bands at K in Ga-doped WS$_2$', fontsize = 14)
    ax.set_xlabel('Ga Concentration (%)', fontsize = 12)
    ax.set_ylabel('Energy (eV)', fontsize = 12)
    ax.set_xticks(arange(numBars))
    ax.set_xticklabels(label_list, fontsize = 12)

    plt.tight_layout()
    if save:
        plt.savefig(outfile)

    plt.show()


def plotPoints(
        ebounds = [-4, 0.75],
        EIGENVAL = 'EIGENVAL',
        DOSCAR = 'DOSCAR',
        kpoint = [1/3, 1/3, 0],
        printInfo = True,
        scale = 'linear',
        save = False,
        outfile = 'eigevalues.pdf',
        ):
    """
    plots eigenvalues for various concentrations
    EIGENVAL: EIGENVAL file (str)
    POSCAR: POSCAR file (str)
    DOSCAR: DOSCAR file (str)
    kpoint: k-point at which eigenvalues are read (list of 3 floats)
    printInfo: if True, prints band gaps and Fermi energies as they are read (bool)
    """
    eigenval_tab = getEigenvalTab(EIGENVAL, DOSCAR, kpoint, printInfo)
    root = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands'
    EIGENVAL = '%s/%s' %(root, EIGENVAL)
    DOSCAR = '%s/%s' %(root, DOSCAR)
    prisEigenval_tab = getEigenvalTab(EIGENVAL, DOSCAR, kpoint, printInfo, walk = False)
    eigenval_tab += prisEigenval_tab

    fig, ax = plt.subplots()
    
    for natoms, eigenval_list in eigenval_tab:
    
        print('natoms = %s' %natoms)
        if natoms > 0:
            con_list = [100/natoms for n in range(len(eigenval_list))] 
        else:
            con_list = [0 for n in range(len(eigenval_list))]

        ax.plot(con_list, eigenval_list, 'bo')

    if scale == 'log':
        ax.set_xscale("log", nonposx='clip')
    ax.set_ylim(ebounds)
    ax.set_title('Eigenvalues at K in Ga-doped WS$_2$', fontsize = 14)
    ax.set_xlabel('Ga concentration (%)', fontsize = 12)
    ax.set_ylabel('eigenvalues (eV)', fontsize = 12)

    if save:
        plt.savefig(outfile)

    plt.show()
        
#------------------------------- SCRATCH ---------------------------------------------
    # get Fermi energies from DOSCARs
#    efermi_list = [getFermiEnergy(DOSCAR) for DOSCAR in DOSCAR_list]

                        # print('dif = %s' %dif)
#        eigenval_dict[EIGENVAL.split('/')[1]] = natoms, eigenval_list

    # find band whose maxima is closest to but less than the fermi energy
#    vbDif = max_list.max() - min_list.min()
#    for en in max_list:
#        newVbDif = efermi - en
#        if newVbDif > 0:
#            vbDif = newVbDif
#            vbm = en
#        else:
#            break
#
#    # find band index that contains vbm
#    vbmBand = max_list.tolist().index(vbm)
        
#def getPrisBars(
#        ebounds = [-4, 1.75],
#        EIGENVAL = 'EIGENVAL',
#        DOSCAR = 'DOSCAR',
#        kpoint = [1/3, 1/3, 0],
#        printInfo = True,
#        thres = 0.3, #eV
#        ):
#    """
#    gets energy ranges for bars representing vbm, cbm, and midgap states
#    EIGENVAL: EIGENVAL file (str)
#    POSCAR: POSCAR file (str)
#    DOSCAR: DOSCAR file (str)
#    kpoint: k-point at which eigenvalues are read (list of 3 floats)
#    printInfo: if True, prints band gaps and Fermi energies as they are read (bool)
#    thres: threshold for gap in eV (float)
#    """
#    root = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands'
##    EIGEVAL = '~/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands/%s' %EIGENVAL
##    EIGEVAL = '~/Desktop/meunier/ionIrrad/tmd/WS2_optb88/bands/%s' %EIGENVAL
#    EIGENVAL = '%s/%s' %(root, EIGENVAL)
#    DOSCAR = '%s/%s' %(root, DOSCAR)
#    eigenval_tab = getEigenvalTab(EIGENVAL, DOSCAR, kpoint, printInfo, walk = False)
#    
#    emin, emax = ebounds
#
#    top_tab = []
#    bot_tab = []
#    natoms_list = []
#    bar_dict = {}
#    for natoms, eigenval_list in eigenval_tab:
##    for natoms in eigenval_tab:
##        eigenval_list = eigenval_tab[natoms]
#
#        # find of energy ranges in which eigenvalues are closely spaced
#        top_list = []
#        bot_list = [emin]
#        for n in range(len(eigenval_list) - 1):
#            thisEig = eigenval_list[n]
#            nextEig = eigenval_list[n + 1]
#            if thisEig > emin and thisEig < emax:
#                if nextEig - thisEig > thres:
#                    top_list.append(thisEig)
#                    bot_list.append(nextEig)
#
#        top_list.append(emax)
#        top_tab.append(top_list)
#        bot_tab.append(bot_list)
#        natoms_list.append(natoms)
#   
#        bar_dict[0] = top_list, bot_list
#
##    return top_tab, bot_tab, natoms_list, bar_dict
#    return bar_dict
#
#

#                       color = barColor, zorder = -1, alpha = .4, edgecolor = barColor)
#                endPointsY = [bandEdge, bandEdge]
#                ax.plot(endPointsX, endPointsY, color = barColor, linewidth = 2)

#                   color = barColor, edgecolor = barColor)

#        elif natoms > 10:
#            label_list.append('%.4s' %(100 / natoms))
#        else:
#            label_list.append('%.3s' %(100 / natoms)) # hide trailing decimal
