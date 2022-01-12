# Anthony Yoshimura
# 01/10/17

# plots the energies from a set of calculations
# requires OUTCARs and POSCARs (for Birch-Murnaghan fit)
# order of operations: enCon, kCon, cellCon, vacCon, layCon

from scipy.optimize import curve_fit, newton
from numpy import linspace, array, round, floor, prod
from numpy.linalg import norm
from inspect import signature
import matplotlib.pyplot as plt
import POSCAR as p
import os
from getGCD import getGCD

# use full numbers (not offset) on energy axis
import matplotlib
matplotlib.rc('axes.formatter', useoffset = False)

def getPaths(name = 'OUTCAR', top = '.', mustContain = None):
    """
    returns a list of all paths to files named 'name'
    name: name of file (str)
    top: top-level directory (str)
    mustContain: name(s) of directory(ies) that must be included in root (str)
    """
    path_list = []
    for root, dirs, files in os.walk(top):
        if name in files:
            if type(mustContain) == str:
                if mustContain in root:
                    path_list.append(os.path.join(root, name))

            elif type(mustContain) == list or type(mustContain) == tuple:
                include = True
                for directory in mustContain:
                    if directory not in root:
                        include = False
                        break
                if include:
                    path_list.append(os.path.join(root, name))

            else:
                path_list.append(os.path.join(root, name))

    path_list.sort()
    return path_list

def getEnergy(infile = 'OUTCAR'):
    """ returns energy from OUTCAR file """
    with open(infile) as f:
        for line in f:
            if 'free  energy' in line:
                energy = float(line.split()[-2])
    return energy

def getVolume(infile = 'POSCAR'):
    """ returns volume from POSCAR file """
    pos = p.POSCAR(infile)
    volume = pos.getCellVolume()
    return volume

def getPressure(infile = 'OUTCAR'):
    """ returns presure in kB from OUTCAR file """
    with open(infile) as f:
        for line in f:
            if 'pressure' in line:
                energy = float(line.split()[3])
    return energy

def getMaxPot(infile = 'LOCPOT'):
    """ returns maximum potential from LOCPOT file """
    with open(infile) as f:
        for n in range(6): # get to populations line
            f.readline()

        numAtoms = sum([int(val) for val in f.readline().split()]) # get number of atoms

        for n in range(numAtoms + 2): # get to grid dimensions
            f.readline() 

        numPts = prod([int(val) for val in f.readline().split()]) # get number of grid points

        pot_list = []
        for line in f:
            for val in line.split():
                pot_list.append(float(val))

#        pot_ar = array(pot_tab)
#        pot_ar.reshape(numPts)
#        maxPot = m

#    return maxPot
    return max(pot_list)
    
def getEnergies(
        infile = 'OUTCAR',
        systemDepth = 1,
        top = '.',
        mustContain = None,
        printPaths = True,
        neb = False,
        ):
    """
    returns table of dictionaries containing the energies of various systems
        * very well suited for NEB!
    systemDepth: depth of directory names to be used as system names (pos int)
    mustContain: name of directory that must be contained in path (str)
    """
    path_list = getPaths(name = infile, top = top, mustContain = mustContain)
    system_dict, energy_dict = {}, {}
    table = [system_dict, energy_dict]

    # set species key to '.' if there are no species to iterate over
#    print(path_list)
#    spec_index = 0

    # -3 works for min path-length
    if len(path_list[0].split('/')) - 3 < systemDepth:
        spec_index = 0
    else:
        spec_index = 1
    if neb:
        spec_index = -4

    if printPaths:
        print('extracting energies from:')
    for path in path_list:
        try:
            energy = getEnergy(path)
            if printPaths:
                print(' %s' %path)
        except UnboundLocalError:
            print('  energy not found in %s' %path)

        path = path.split('/')
        spec = path[spec_index]
        if spec not in system_dict:
            system_dict[spec], energy_dict[spec] = [], []
        system = path[-1 - systemDepth]
        system_dict[spec] += [system]
        energy_dict[spec] += [energy]

    return table

def getVolumes(
        infile = 'POSCAR',
        systemDepth = 1,
        top = '.',
        mustContain = None,
        ):
    """
    returns table of dictionaries containing the volumes of various systems
    systemDepth: depth of directory names to be used as system names (pos int)
    mustContain: name of directory that must be contained in path (str)
    """
    path_list = getPaths(name = infile, top = top, mustContain = mustContain)
    system_dict, volume_dict = {}, {}
    table = [system_dict, volume_dict]

    # set species key to '.' if there are no species to iterate over
    if len(path_list[0].split('/')) - 3 < systemDepth:
        spec_index = 0
    else:
        spec_index = 1

    print('extracting volumes from:')
    for path in path_list:
        try:
            volume = getVolume(path)
            print(' %s' %path)
        except UnboundLocalError:
            print('  POSCAR not found in %s' %path)
        path = path.split('/')
        spec = path[spec_index]
        if spec not in system_dict:
            system_dict[spec], volume_dict[spec] = [], []
        system = path[-1 - systemDepth]
        system_dict[spec] += [system]
        volume_dict[spec] += [volume]

    return table

def getMaxPots(
        infile = 'LOCPOT',
        systemDepth = 1,
        top = '.',
        mustContain = None,
        ):
    """
    returns table of dictionaries containing the max potential energy of
        various systems
    systemDepth: depth of directory names to be used as system names (pos int)
    mustContain: name of directory that must be contained in path (str)
    """
    path_list = getPaths(name = infile, top = top, mustContain = mustContain)
    system_dict, volume_dict = {}, {}
    table = [system_dict, volume_dict]

    # set species key to '.' if there are no species to iterate over
    if len(path_list[0].split('/')) - 3 < systemDepth:
        spec_index = 0
    else:
        spec_index = 1

    print('extracting potentials from:')
    for path in path_list:
        try:
            maxPot = getMaxPot(path)
            print(' %s' %path)
        except UnboundLocalError:
            print('  LOCPOT not found in %s' %path)
        path = path.split('/')
        spec = path[spec_index]
        if spec not in system_dict:
            system_dict[spec] += [system]
            volume_dict[spec] += [maxPot]

    return table


#--------------------------- Plot Convergence -------------------------------
def plot(
        e_list = 'auto',
        s_list = 'auto',
        infile = 'OUTCAR',
        outfile = 'convergence.pdf',
        mustContain = None,
        systemDepth = 1,
        title   = 'Energy Convergence',
        xlabel  = 'System',
        ylabel  = 'Energy (eV)',
        ybounds = None,
        xbounds = None,
        showConvergence = False,
        thres   = .001,
        surface = False,
        supercell = False,
        constant_volume = True,
        ebulk   = -8.0,
        eps_pris = 0,
        eps_dict = None,
        ticks = 'auto',
        labels = 'auto',
        align = False,
        scale = 1,
        ):
    """
    plots energies with respect to system
    e_list: list of energies in eV (list of floats)
    s_list: list of systems corresponding to energies (list of str)
    outfile: file to which plot will be saved (str)
    systemDepth: depth of directory names to be used as system names (pos int)
    mustContain: name of directory that must be contained in path (str)
    title, xlabel, ylabel: title and axes labels (str)
    xbounds, ybounds: max width and height (tuples of 2 floats)
    surface: if True, surface energies are plotted (bool)
    supercell: if True, formation energies are plotted vs. supoercell dimensions (vool)
    constant_volume: if False, energies are normalized to the volume
        of the smallest system (bool)
    ebulk: energy of pristine bulk cell in eV (float)
    eps_pris: energy of atoms gained or lost in defect (float)
    thres: theshold of convergence in eV (float)
    align: if True, sets all first energies equal to 0 (bool)
    labels: dict of labels to be used in legend (dict of str)
    ticks: tick labels (list of str)
    """
    # get energies from vasp run directories
    if e_list == 'auto':
        e_dict = getEnergies(infile, systemDepth, mustContain = mustContain)[1]
    else:
        e_dict = {'.': e_list}
        print('energies:', e_dict)

    # get names of systems for labelling
    if s_list == 'auto':
        s_dict = getEnergies(infile, systemDepth, mustContain = mustContain)[0]
    else:
        s_dict = {'.': s_list}
        print('systems:', s_dict)

    print('e_dict: %s\ns_dict: %s' %(e_dict, s_dict))
    # remove uncessary leading zeroes
    int_list = s_dict['.']
    try:
        int_list = [int(val) for val in int_list]
        s_list = [str(val) for val in int_list]
        s_dict['.'] = s_list
    except ValueError:
        pass

    # get volumes of systems
    if constant_volume == False:
        v_dict = getVolumes(systemDepth, mustContain)[1]

    # plotting
    fig, ax = plt.subplots()
    if align:
        ax.axhline(linestyle = 'dashed')
    for spec in e_dict:
        e_ar = array(e_dict[spec])
        s_list = s_dict[spec]

        # align all curves at start
        if align:
            e_ar = e_ar - e_ar[0]

        # add chemical potential of defect gas
        if eps_dict != None:
            e_ar += eps_pris
        
        # surface energy
        if surface:
            new_list = []
            for energy, system in zip(e_ar, s_list):
                system = float(system)
                surf = 0.5 * (energy - system*ebulk)
                new_list.append(surf)
            e_ar = new_list

        # volume proportionality
        if not constant_volume:
            v_list = v_dict[spec]
            prop_ar = array(v_list) / v_list[0]

            # substract pristine lattice of appropriate size with defect gas
            bulk_list = array([ebulk * prop + eps_dict[spec] for prop in prop_ar])
            print(bulk_list)
            e_ar -= bulk_list
            e_ar /= prop_ar * scale

        # plot formation energy vs. supercell dimensions
        if supercell:
            print(spec_list)

        if labels == 'auto':
            ax.plot(e_ar, label = spec, linewidth = 2)
        else:
            ax.plot(e_ar, label = labels[spec], linewidth = 2)

    # ticks
    if ticks == 'auto':
        print(e_ar)
        ticks = [n for n in range(len(e_ar))]
    if len(s_list[0]) > 5:
        ax.set_xticklabels(s_list, rotation = 'vertical')
    else:
        ax.set_xticklabels(s_list)

    # convergence annotation: find n such that delta E_n < threshold
    if showConvergence:
        convergence = None

        # find all energy differences between current point and remaining points
        for n in range(len(e_ar) - 1):
            diff_list = [abs(e_ar[n] - e_ar[n + m + 1]) for m in range(len(e_ar) - n - 1)]
            converged = True
            for diff in diff_list:
                if diff > thres:
                    converged = False
                    print(diff)
                    break
            if converged:
                convergence = n
                break

        mid = len(e_ar) // 2
        concavity = e_ar[0] + e_ar[-1] - 2 * e_ar[mid]  # for annotation arrow
        if concavity > 0:
            offset = 40
        else:
            offset = -40

        if convergence != None:
            thres *= 1000
            ax.annotate(r'$\Delta E \leq$ %s meV' %thres,
                        fontsize = 14, ha = 'center',
                        xy = (n, e_ar[n]), xycoords = 'data',
                        xytext = (0, offset), textcoords = 'offset points',
                        arrowprops = dict(facecolor = 'k', shrink = 0.05))

    # plot attributes
    ax.grid()
    if len(e_dict) > 1:
        ax.legend(loc = 1)
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_xticks(ticks)
    ax.set_xlim(0, len(e_ar) - 1)
    if ybounds != None:
        ax.set_ylim(ybounds)

    plt.tight_layout()
    if type(outfile) == str:
        plt.savefig(outfile)
    plt.show()


#------------------------ Birch-Murnaghan Curve Fit ---------------------------
def bM(v, e0, b0, bp, v0):
    """
    returns energy in eV predicted by Birch-Murnaghan EOS (PRB 70, 224107)
    """
    eta = (v0/v)**(2/3)
    E = e0 + 9*b0*v0/16 * (eta - 1)**2 * (6 - 4*eta + bp*(eta - 1))
    return E

def pressure(v, b0, bp, v0):
    pass

def parabola(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

def fit(v_list = 'auto',
        e_list = 'auto',
        guess = 'auto',
        function = 'bM',
        plot = True,
        outfile = 'fit.pdf',
        system = 'auto',
        layer = False,
        top = '.',
        destination = '.',
        title = 'auto',
        save = True,
        ):
    """
    returns: dictionary of optimized parameters and 
        plots best fit for given energy and volume data
    system: chemical formula for system (str)
        * if auto: chemical formula deduced from POSCARs
    e_list / v_list: list of energies / volume in eV / Å^3 (list of floats)
    guess: initial guess of fit parameters [E0, B0, B', V0] (list of 4 floats)
    function: type of curve to be fit (str)
    outfile: name of plot file (str)
    """
    # UNDER CONSTRUCTION:
    #    automatically detect if layer
    #    handle path keys better

    # obtain data from VASP run
    if top == '.':
        key = top.split('/')[0]
    else:
        key = top.split('/')[1]

    if v_list == 'auto':
        v_list = getVolumes(top = top)[1][key]
    if e_list == 'auto':
        e_list = getEnergies(top = top)[1][key]

    # initial guess of parameters (e0, b0, bp, v0)
    e_min = min(e_list)
    v_min = v_list[e_list.index(e_min)]
    if guess == 'auto':
        guess = (e_min, .02, 20, v_min)

    # BM fit
    if 'b' in function or 'B' in function:
        print('fitting Birch-Murnaghan function')
    
        # find curve_fit parameters
        e0, b0, bp, v0 = curve_fit(bM, v_list, e_list, p0 = guess)[0]
        print(
            "fitted parameters:\n\tE0 = %.4g, B0 = %.4g, B' = %.4g, V0 = %.4g"
                %(e0, b0, bp, v0))

        # coefficient of determination
        mean = sum(e_list) / len(e_list)
        ss_tot = sum( [(e - mean)**2 for e in e_list] )
        ss_res = sum( [(e_list[n] - bM(v_list[n], e0, b0, bp, v0) )**2
                     for n in range(len(e_list))] )
        R = 1 - ss_res / ss_tot
        print("coefficient of determination:\n    R = %.10g" %R) 
    
    # parabolic fit
    elif 'p' in function or 'P' in function:
        print('fitting parabola')

        # find curve_fit parameters
        p1, p2, p3 = curve_fit(parabola, v_list, e_list)[0]
        v0 = -p2/(2*p1)
        e0 = p3 - p1*v0**2
        print("fitted parameters:\n\tp1 = %.4g, p2 = %.4g, p3 = %.4g"
                %(p1, p2, p3))

        # coefficient of determination
        mean = sum(e_list) / len(e_list)
        ss_tot = sum( [(e - mean)**2 for e in e_list] )
        ss_res = sum( [(e_list[n] - parabola(v_list[n], p1, p2, p3) )**2
                     for n in range(len(e_list))] )
        R = 1 - ss_res / ss_tot
        print("coefficient of determination:\n    R = %.10g" %R) 

    else:
        print('unrecognized fit function. Please set function = bM or parabola.')
        return

    # get lattice constants
    # use POSCAR corresponding to lowest energy system
    paths = getPaths('POSCAR', top = top)
    guess_path = paths[e_list.index(e_min)]
    guess_POSCAR = p.POSCAR(guess_path)

    # scale POSCAR to match optimal volume
    guess_vol = guess_POSCAR.getCellVolume()
    vol_ratio = v0 / guess_vol

    # if POSCAR is 2D system
    if layer:
        lat_ratio = vol_ratio ** (1/2)
        guess_POSCAR.cell[0] *= lat_ratio
        guess_POSCAR.cell[1] *= lat_ratio
        a, b, c = [norm(guess_POSCAR.cell[n]) for n in range(3)]
        print("lattice constants:\n\ta = %.5g, b = %.5g" %(a, b))

    # if POSCAR is 3D system
    else:
        lat_ratio = vol_ratio ** (1/3)
        guess_POSCAR.cell *= lat_ratio
        a, b, c = [norm(guess_POSCAR.cell[n]) for n in range(3)]
        print("lattice constants:\n\ta = %.4g, b = %.4g, c = %.4g" %(a, b, c))

    # plotting
    if plot:

        # define boundaries and get domain
        lower = min(v_list) - 1
        upper = max(v_list) + 1
        v = linspace(lower, upper, 100)
    
        # create figure and axes
        fig, ax = plt.subplots()

        # create title with chemical formula
        specs, pops = guess_POSCAR.specs, guess_POSCAR.pops
        if system == 'auto':
           GCD = getGCD(pops)
           subscripts = [int(pop / GCD) for pop in pops]
           system = ''
           for spec, subscript in zip(specs, subscripts):
               if subscript > 1:
                   system += '%s$_{%s}$' %(spec, subscript)
               else:
                   system += spec

        # plot based on curve fitted parameters
        if 'b' in function or 'B' in function:
            E = bM(v, e0, b0, bp, v0)
            ax.set_title('%s Birch-Murnaghan curve fit' %system, fontsize = 18)

        elif 'p' in function or 'P' in function:
            E = parabola(v, p1, p2, p3)
            if title == 'auto':
                ax.set_title('%s parabolic curve fit' %system, fontsize = 18)
            else:
                ax.set_title(title, fontsize = 18)
            prisV, prisE = v_list[1], e_list[1]
            offset = (upper - lower) * .01
            if prisV < v0:
                offset *= -1
            ax.text(prisV + offset, prisE, 'pristine',
                    transform = ax.transData, fontsize = 12, va = 'top',
                    color = 'red')

        # plot data
        ax.plot(v_list, e_list, 'ro', label = "Data")   # data points
        ax.plot(v, E, label = "R = %.7g" %R)            # best fit

        # labels and formatting
        ax.set_ylabel('Energy (eV)', fontsize = 14)
        ax.set_xlabel(r'Cell Volume (Å$^3$)', fontsize = 14)
        ax.set_xlim(lower, upper)
        ax.grid()

        # text and annotations
        if layer:
            ax.text(.01, .9,
                "Optimal lattice constants:\na = %.5g Å, b = %.5g Å" %(a, b),
                transform = ax.transAxes, fontsize = 12)
        else:
            ax.text(.01, .9,
                "Optimal lattice constants:\na = %.4g, b = %.4g, c = %.4g"
                %(a, b, c), transform = ax.transAxes, fontsize = 12)

        # arrow showing minimum energy / optimal volume
        ax.annotate("E = %.6g eV\nV = %.6g Å$^3$" %(e0,v0),
                    fontsize = 12, ha = 'center',
                    xy = (v0, e0), xycoords = 'data',
                    xytext = (0, 40), textcoords = 'offset points',
                    arrowprops = dict(facecolor = 'k', shrink = 0.05))

        # legend
        ax.legend(loc = 1, fontsize = 12)

        # save and show figure
        plt.tight_layout()
        if save:
            plt.savefig(destination + '/' + outfile)
        plt.show()

    # return info
    if function == 'bM':
        return {'V0': v0, 'E0': e0, 'B0': b0, "B'": bp, 'constants': [a, b, c]}

    elif function == 'parabola':
        return {'a': a, 'b': b, 'c': c, 'constants': [a1, a2, a3]}


def makeOptPOSCAR(
        v_list = 'auto',
        e_ar = 'auto',
        function = bM,
        write = False,
        layer = False,
        top = '.',
        ):
    """
    returns optimized POSCAR corresponding to energy minimum on BM curve.
        * Assumes all directories are named in order of ascending cell size
        * Assumes direct coordinates
    """
    # get correct key for volume and energy dicts
    path_list = top.split('/')
    if len(path_list) > 1:
#        key = path_list[-4]
        key = path_list[1]
    else:
        key = top

    # get all volumes, energies, and paths
    v_list = getVolumes(top = top)[1][key]
    e_list = getEnergies(top = top)[1][key]
    p_list = getPaths('POSCAR', top = top)

    # optimal volume from BM fit
    optv = fit(layer = layer, top = top, save = False)['V0']

    # find POSCARs whose cell volume is adjacent to opt_v
    for n in range(len(v_list)):
        v = v_list[n]
        if optv < v:
            break

    # obtain adjacent POSCARs and their volumes
    gIndex = n
    lIndex = n - 1
    gv = v_list[gIndex]
    lv = v_list[lIndex]
    gPOSCAR = p.POSCAR(p_list[gIndex])
    lPOSCAR = p.POSCAR(p_list[lIndex])

    # linearly interpolate between adjacent POSCARs
    if layer:
        exp = 1/2
    else:
        exp = 1/3
    ll, optl, gl = lv**exp, optv**exp, gv**exp
    ldif = optl - ll
    gdif = gl - ll
    ratio = ldif / gdif

    # interpolate cell
    optPOSCAR = lPOSCAR.copy()
    optPOSCAR.cell = lPOSCAR.cell*(1 - ratio) + gPOSCAR.cell*ratio

    # interpolate coordinates (direct)
    # ensure that linear interpolation occurs between nearest PBC images
    lcoords, gcoords = array(lPOSCAR.coords), array(gPOSCAR.coords)
    difcoords = round(gcoords - lcoords) # account for periodic boundaries
    lcoords += difcoords
    optcoords = lcoords*(1 - ratio) + gcoords*ratio
    optcoords -= floor(optcoords) # bring atoms into cell
    optPOSCAR.coords = optcoords.tolist()

    if write:
        opt_POSCAR.write()

    return optPOSCAR


#------------------------------- fit arbitrary --------------------------------

def fitFunction(function, x_list, y_list, plot = True):
    """
    plots best fit for a given function
    function: function of the form function(x, *args)
    """
    # parameter names and values
    param_list = curve_fit(function, x_list, y_list)[0]
    param_names = str(signature(function)).strip('()').split(', ')
    param_str = 'fitted parameters:\n    ' 

    for n in range(len(param_list)):
        param_str += '%s = %.4g   ' %(param_names[n + 1], param_list[n])
    print(param_str)

    # new function that accepts a list of arguments
    def argsFunction(x, args, f = function):
        return f(x, *args)

    # coefficient of determination
    mean = sum(y_list) / len(y_list)
    ss_tot = sum( [(y - mean)**2 for y in y_list] )
    ss_res = sum( [(y_list[n] - argsFunction(x_list[n], param_list) )**2
                 for n in range(len(y_list))] )
    R = 1 - ss_res / ss_tot
    print("coefficient of determination:\n    R = %.10g" %R) 

    if plot:
        fig, ax = plt.subplots()
        
        ax.plot(x_list, y_list, 'ro', label = 'data')       # data
        lower = min(x_list) - .5
        upper = max(x_list) + .5
        x = linspace(lower, upper, 100)
        y = argsFunction(x, param_list)
        ax.set_xlim(lower, upper)
        ax.plot(x, y, linewidth = 2, label = 'R = %.7g' %R) # fit
        ax.legend(loc = 'best')
        ax.grid()

        # text
        ax.set_title('%s least squares fit' %function.__name__, fontsize = 20)
        ax.set_xlabel('x-data', fontsize = 16)
        ax.set_ylabel('y-data', fontsize = 16)

        ax.text(.01, .90, param_str,
                transform = ax.transAxes, fontsize = 16)

        plt.show()


#--------------------------- test fit with parabola -------------------------------
def fitParabola(x_list, y_list, plot = True):
    a, b, c = curve_fit(parabola, x_list, y_list)[0]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x_list, y_list, 'ro')
        lower = min(x_list) - 1
        upper = max(x_list) + 1
        x = linspace(lower, upper, 100)
        y = parabola(x, a, b, c)
        ax.plot(x, y)
        plt.show()

    return a, b, c

def parabolaPrime(x, a, b, c):
    dy = 2*a*x + b
    return dy

def parabolaPrimePrime(x, a, b, c):
    d2y = 2*a
    return d2y

def findPMin(x_list, y_list):
    params = fitParabola(x_list, y_list, plot = False)
    x0 = x_list[len(x_list) // 2]
    min_x = newton(parabolaPrime, x0, parabolaPrimePrime, params)
    return min_x

def cubic(x, a, b, c, d):
    y = a*x**3 + b*x**2 + c*x + d
    return y

def cubicPrime(x, a, b, c, d):
    dy = 3*a*x**2 + 2*b*x + c
    return dy

def cubicPrimePrime(x, a, b, c, d):
    d2y = 6*a*x + 2*b
    return d2y

def plotCubic(x_list, y_list):
    a, b, c, d = curve_fit(cubic, x_list, y_list)[0]

    fig, ax = plt.subplots()
    ax.plot(x_list, y_list, 'ro')
    lower = min(x_list) - 1
    upper = max(x_list) + 1
    x = linspace(lower, upper, 100)
    y = cubic(x, a, b, c, d)
    yp = cubicPrime(x, a, b, c, d)
    ypp = cubicPrimePrime(x, a, b, c, d)
    ax.plot(x, y, label = 'y')
#    ax.plot(x, yp, label = "y'")
#    ax.plot(x, ypp, label = "y''")
    ax.grid()
#    ax.axhline(y = 0)
    ax.legend(loc = 1)
    plt.show()

def findCMin(x_list, y_list, plot = False):
    a, b, c, d = curve_fit(cubic, x_list, y_list)[0]
    print(a, b, c, d)
    x0 = x_list[len(x_list) // 2]
    min_x = newton(cubicPrime, x0, cubicPrimePrime, args = (a, b, c, d))

    if plot:
        plotCubic(x_list, y_list)

    return min_x


#---------------------------------- scratch ------------------------------------
def plotBM(v_list, e_ar):
    e0, b0, bp, v0 = curve_fit(bM, v_list, e_ar, maxfev = 10000)[0]
    print(e0, b0, bp, v0)

    fig, ax = plt.subplots()
    ax.plot(v_list, e_ar, 'ro')

    lower = min(v_list) - 1
    upper = max(v_list) + 1
    v = linspace(lower, upper, 100)
    E = bM(v, e0, b0, bp, v0)
    Ep = bMPrime(v, e0, b0, bp, v0)
    ax.plot(v, E, label = "E")
#    ax.plot(v, Ep, label = "E'")
    ax.grid()
#    ax.axhline(y = 0)
    ax.legend(loc = 1)
    plt.savefig('temp.pdf')
    plt.show()
    
def bMPrime(v, e0, b0, bp, v0):
#    eta = (v0/v)**(1/3)
#    dE = 9*b0*v0/16 * (-2*bp*eta**5/v0*(eta**2 - 1)**2 + 16/3/v0*eta**7 - 20/3/v0*eta**5)
    eta = (v/v0)**(1/3)
    dE = 9*b0*v0/16 * (eta**2 - 1) * (2/3/v0/eta) * (16 - 8*eta**2 + bp*eta**2)
    return dE 

def bmPrimePrime(v, e0, b0, bp, v0):
    pass

def objective(v, E, e0, b0, bp, v0):
    err = E - bM(v, e0, b0, bp, v0)
    return err

def findBMMin(v_list, e_ar, plot = False):
#    e0, b0, bp, v0 = fitBirchMurnaghan(v_list, e_ar, plot = False)
    e0, b0, bp, v0 = curve_fit(bM, x_list, y_list)[0]
#    f = bM(v, e0, b0, bp, v0)
#    fp = prime(v, e0, b0, bp, v0)
    x0 = v_list[len(v_list) // 2]
#    min_v = newton(f, x0, fp)
#    min_v = newton(bM, x0, fprime = prime, args = params)
    min_v = newton(bM, x0, args = (e0, b0, bp, v0))

    if plot:
        plotBM(v_list, e_ar)

    return min_v

#        gca().getxaxis().get_major_formatter().set_useOffset(False)
