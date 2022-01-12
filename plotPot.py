# plots potential energy as a function of distance

# Anthony Yoshimura
# 02/09/18

from scipy.optimize import curve_fit
from numpy import array, linspace, dot, exp
from numpy.linalg import norm
from gaussxw import gaussxwab
from plotEnCon import getEnergy, getEnergies
from os import walk
import POSCAR as p
import plotEnCon as pe
import matplotlib.pyplot as plt
#from ionIrrad import getIncAndTarg

def getLJ(r, eps, rm):
    """
    Returns Lennard-Jones potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    """
    return eps * ((rm/(r))**12 - 2*(rm/r)**6)


def getPower(r, eps, rm, a = 8, b = 6):
    """
    Returns Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    {a,b}: exponents on first and second terms (pos ints)
    """
    c1 = b / (a - b)
    c2 = a / (a - b)
    return eps*(c1*(rm/r)**a - c2*(rm/r)**b)


def getInv(r, a, b):
    """
    returns inverse power type potential at position r
    a: multiplicative constant (pos float)
    b: exponent (pos float)
    """
    return a / r**b


def getExp(r, eps, rm, a = 1, b = 6):
    """
    Returns Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    a: constant muliplying exponent on first term (float)
    b: exponents on second term (pos int)
    """
    c1 = b / (a - b)
    c2 = a / (a - b)
    return eps*(c1*exp(a*(1 - r/rm)) - c2*(rm/r)**b)


def getPowerPrime(r, eps, rm, a, b):
    """
    Returns derivative of Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    {a,b}: exponents on first and second terms (pos ints)
    """
    c = eps * a * b / (a - b) / rm
    return c * ((rm / r)**(b + 1) - (rm / r)**(a + 1))


def getPosEn(top = '.', incident = 13, source = 1):
    """
    returns energy and position arrays from vasp runs
    top: directory that contains vasp run directories (str)
    """
    # list directories
    dir_list = list(next(walk('.'))[1])
    dir_list.sort()

    pos_list = [float(val) for val in dir_list if 'n' not in val]
    neg_list = [val for val in dir_list if 'n' in val]
    neg_list.reverse()
    neg_list = [-float(val[1:]) for val in neg_list]

    dist_list = []
    for directory in dir_list:
        with open('%s/POSCAR' %directory) as f:
            # Get cell and species info
            f.readline() #skip comment line
            scale = float(f.readline())
            cell = [ [float(val) for val in f.readline().split()] for n in range(3)]
            cellHeight = cell[2][2]
            specs = f.readline().split()
            pops = [int(val) for val in f.readline().split()]
            numAtoms = sum(pops)
        
            # assume extra line from selective dynamics
            for n in range(source + 1):
                f.readline()
            pos1 = array([float(val) for val in f.readline().split()[:3]])
            for n in range(incident - source - 1):
                f.readline()
            pos2 = array([float(val) for val in f.readline().split()[:3]])
            dif = pos1 - pos2
            dif = dot(dif, cell)
            dist = norm(dif)

    pos_list = [float(val) for val in dir_list if 'n' not in val]
    neg_list = [val for val in dir_list if 'n' in val]
    neg_list.reverse()
    neg_list = [-float(val[1:]) for val in neg_list]
    dist_list = neg_list + pos_list

    en_list = []
    for dist in dist_list:
        if dist < 0:
            en = getEnergy('n%s/OUTCAR' %(-dist))
        else:
            en = getEnergy('%s/OUTCAR' %dist)
        en_list.append(en)

    dist_list = (array(dist_list) + bondlength).tolist()
    en_list = (array(en_list) - en_list[-1]).tolist()

    return dist_list, en_list


def getMDPosEn(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        OUTCAR = 'OUTCAR',
        incAtom = 76,
        targAtom = 12,
        ref = None,
        ):
    """
    returns lists of energies (eV) and positions (Angstrom)
    XDATCAR: XDATCAR file containing incAtom positions (str)
    OSZICAR: OSZICAR file containing free energies (str)
    incAtom: incAtom atom number (starting from 1) whose position is tracked (pos int)
        * if 'auto', incAtom with most movement is chosen
    soure: targAtom atom number (starting from 1) whose position is tracked (pos int)
        * if 'auto', targAtom is chosen as closest to incAtom axis
    direction: direction in which incAtom is moving (1, 2, or 3)
    ref: reference energy set equal to zero (eV)
        * if None, ref set to first energy in OSZICAR
        * if 'auto', energy is chosen from static VASP run
    """
    # automatically get inc and targ atoms
    if incAtom == 'auto' and targAtom == 'auto':
        incAtom, targAtom, minDist = getIncAndTarg(XDATCAR) # indices starting from 0
        incAtom += 1
        targAtom += 1

    # order inc and targ atoms
    first = targAtom  # index of first relevant atom
    if targAtom > incAtom:
        first = incAtom
    second = abs(incAtom - targAtom) 

    # read XDATCAR
    with open(XDATCAR) as f:

        # Get cell and species info
        f.readline() #skip comment line
        scale = float(f.readline())
        cell = [ [float(val) for val in f.readline().split()] for n in range(3)]
        cellHeight = cell[2][2]
        specs = f.readline().split()
        pops = [int(val) for val in f.readline().split()]
        numAtoms = sum(pops)

        # collect distances between incAtom and targAtom atoms
        dist_list = []

        # loop through XDATCAR configurations
        while True:

            # skip to first relevant atom
            for n in range(first):
                f.readline()
            pos1 = array([float(val) for val in f.readline().split()])

            # break from while loop when position data ends
            if len(pos1) < 3:
                print('reached end of XDATCAR file')
                break

            for n in range(second - 1): # skip to second relevant atom
                f.readline()
            pos2 = array([float(val) for val in f.readline().split()])

            # get distance between atoms
            dif = pos2 - pos1
            dif = dot(dif, cell)
            dist_list.append(norm(dif))

            # skip to next set of coords
            for n in range(numAtoms - first - second):
                f.readline()

    # get energies from OSZICAR
    en_list = []
    with open(OSZICAR) as f:
        for line in f:
            if 'F=' in line:
                line_list = line.split()
                en_list.append(float(line_list[line_list.index('F=') + 1]))

    # substract ref energy from energies
    if ref == None:
        ref = en_list[0]

    elif ref == 'auto':
        pos = p.getFromXDAT(1, XDATCAR, OUTCAR = OUTCAR)
        targSpec = pos.getSpecOf(targAtom)
        root = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/MD/collision/%sS2/charged/467_333' %targSpec
        ref = pe.getEnergy('%s/OUTCAR' %root)

    en_list = (array(en_list) - ref).tolist()
            
    return dist_list, en_list


def getGaussPosEn(top = '.'):
    """
    returns enery and position arrays from vasp runs
    top: directory that contains vasp run directories (str)
    """
    x_ar, w_ar = gaussxwab(20, .55, 1)
    x_ar.sort()
    U_ar = (getEnergies(printPaths = False)[1]['.'])
    U_ar = (array(U_ar) - U_ar[-1]).tolist()
    return x_ar, U_ar


def plot(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        calculationType = 'MD',
        incAtom = 76,     # energy with respect to interatomic distance
        targAtom = 12,
        title = 'auto',
        save = False,
        outfile = 'lj.pdf',
        grid = True,
        fit_list = ['inv'],
        ref = None,
        ebounds = None,
        rbounds = None,
        catch = False,
        ):
    """
    plots energy vs position data with curve fit
    XDATCAR: XDATCAR file containing atom positions (str)
    OSZICAR: OSZICAR file containing free energies (str)
    calculationType: type of vasp calculations used to find U ('Norm', 'MD', or 'gauss')
    incAtom: atom number (starting from 1) whose position is tracked (pos int)
        * if incAtom = 'auto', atom with most movement is chosen
    fit_list: list of curve fitting types (list of str)
    ref: reference for noninteracting (zero) energy in eV (float)
    """
    if calculationType[0] == 'm' or calculationType[0] == 'M':
        dist_list, en_list = getMDPosEn(XDATCAR, OSZICAR, incAtom, targAtom, ref)
    elif calculationType[0] == 'g' or calculationType[0] == 'G':
        dist_list, en_list = getGaussPosEn(incAtom = incAtom, targAtom = targAtom)
    else:
        dist_list, en_list = getPosEn(incAtom = incAtom, targAtom = targAtom)

    # guess initial parameters (eps, rm)
    en_min = min(en_list) - en_list[0]
    r_min = dist_list[en_list.index(min(en_list))]
    guess = (-en_min, r_min, 12, 6)


    # position space over which to plot
    rmin, rmax = min(dist_list), max(dist_list)
    domain = rmax - rmin
    buff = .05 * domain
    lower, upper = rmin - buff, rmax + buff
    r = linspace(rmin, rmax, 200)

    # prepare figure
    fig, ax = plt.subplots()

    # curve fits
    for fit in fit_list:

        # Lennard-Jones-power fit
        if fit[0] == 'p' or fit[0] == 'P':
            print('fitting Lennard-Jones-power potential')
            peps, prm, pa, pb = curve_fit(getPower, dist_list, en_list, p0 = guess)[0]
            print('power fitted parameters:')
            print('    epsilon = %.4g eV, rm = %.4g A, a = %.4g, b = %.4g' %(peps, prm, pa, pb))
    
            # coefficient of determination for power
            mean = sum(en_list) / len(en_list)
            ss_tot = sum( [(en - mean)**2 for en in en_list] )
            ss_res = sum( [(en - getPower(r, peps, prm, pa, pb))**2 for en, r in zip(en_list, dist_list)] )
            PR = 1 - ss_res / ss_tot
            print('coefficient of determination for power fit:\n    R = %.10g' %PR)

            # fitted curve for plotting
            Power = getPower(r, peps, prm, pa, pb)
            ax.plot(r, Power, label = 'Pow: R = %.4g' %PR, zorder = 3)
            ax.text(.99, .99, r'$\epsilon$ = %.3g eV, $r_m$ = %.3g $\AA$, $a$ = %.3g, $b$ = %.3g'
                      %(peps, prm, pa, pb), fontsize = 12, ha = 'right', va = 'top',
                      transform = ax.transAxes, color = 'tab:blue')
    
        # Lennard-Jones-exponential fit
        if fit[0] == 'e' or fit[0] == 'E':
            print('fitting Lennard-Jones-exp potential')
            eeps, erm, ea, eb = curve_fit(getExp, dist_list, en_list, p0 = guess)[0]
            print('exp fitted parameters:')
            print('    epsilon = %.4g eV, rm = %.4g A, a = %.4g, b = %.4g' %(eeps, erm, ea, eb))
    
            # coefficient of determination for exp
            mean = sum(en_list) / len(en_list)
            ss_tot = sum( [(en - mean)**2 for en in en_list] )
            ss_res = sum( [(en - getExp(r, eeps, erm, ea, eb))**2 for en, r in zip(en_list, dist_list)] )
            ER = 1 - ss_res / ss_tot
            print('coefficient of determination for exp fit:\n    R = %.10g' %ER)

            # fitted curve for plotting
            Exp = getExp(r, eeps, erm , ea, eb)
            ax.plot(r, Exp, label = 'Exp: R = %.4g' %ER, zorder = 3)
            ax.text(.99, .92, r'$\epsilon$ = %.3g eV, $r_m$ = %.3g $\AA$, $a$ = %.3g, $b$ = %.3g'
                      %(eeps, erm, ea, eb), fontsize = 12, ha = 'right', va = 'top',
                      transform = ax.transAxes, color = 'tab:orange')
    
        # inverse power fit
        if fit[0] == 'i' or fit[0] == 'I':
            print('fitting inverse power potential')
            ia, ib = curve_fit(getInv, dist_list, en_list, p0 = (60, 2))[0]
            print('exp fitted parameters:')
            print('    a = %.4g, b = %.4g' %(ia, ib))
    
            # coefficient of determination for inverse power
            mean = sum(en_list) / len(en_list)
            ss_tot = sum( [(en - mean)**2 for en in en_list] )
            ss_res = sum( [(en - getInv(r, ia, ib))**2 for en, r in zip(en_list, dist_list)] )
            IR = 1 - ss_res / ss_tot
            print('coefficient of determination for inverse fit:\n    R = %.10g' %IR)

            # fitted curve for plotting
            Inv = getInv(r, ia, ib)
            ax.plot(r, Inv, label = 'Inv: R = %.4g' %IR, zorder = 3)
            ax.text(.99, .99, r'$a$ = %.3g, $b$ = %.3g'
                      %(ia, ib), fontsize = 12, ha = 'right', va = 'top',
                      transform = ax.transAxes, color = 'tab:orange')

            # return fitted parameters
            if catch:
                return ia, ib
    
#    polyTest = getPower(array(dist_list), guess[0], guess[1] + .01, a = 1)
#    expTest = getPowerxp(array(dist_list), guess[0], guess[1], a = 3)
    

    # plot data and fit
    ax.plot(dist_list, en_list, 'ro', label = 'Data', zorder = 2)
#    ax.plot(dist_list, en_list, label = 'Data', zorder = 2)
#    ax.plot(dist_list, polyTest, label = 'polyTest')
#    ax.plot(dist_list, expTest, label = 'expTest')
#    print('en_min = %s, r_min = %s' %guess)
#    print(expTest)

    # title
    if title == 'auto':
        title = 'Ga - W interaction potential (%s)' %calculationType

    # plot formatting
    ax.set_xlim(0, upper)
    if ebounds == None:
        ebounds = min(en_list) - 1, max(en_list) + 1
    ax.set_ylim(ebounds)
    ax.set_title(title, fontsize = 14)
    ax.set_xlabel(r'position ($\AA$)', fontsize = 12)
    ax.set_ylabel('energy (eV)', fontsize = 12)
#    ax.legend(loc = 'best')
#    ax.legend(loc = (.66, .67))
    ax.legend(loc = (.66, .80))
    ax.axhline(y = 0, color = 'k', ls = '--', zorder = 2)
#    ax.grid(zorder = -4)
    if grid:
        ax.grid()

    # text
#    ax.text(.03, .99, '$\epsilon$ = %.3g eV, $r_m$ = %.3g $\AA$, $a$ = %.3g, $b$ = %.3g'
#          %(peps, prm, pa, pb),
#           fontsize = 11, ha = 'left', va = 'top', transform = ax.transAxes, color = 'tab:blue')

    if save:
        plt.savefig(outfile)
    plt.show()

