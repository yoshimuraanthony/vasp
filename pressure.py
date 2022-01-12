# Anthony Yoshimura
# 10/23/18

import POSCAR as p
from plotEnCon import getPaths
import os

from scipy.optimize import leastsq
from scipy.optimize import fsolve

from numpy import linspace, array, round, floor, mgrid, ravel
from numpy import cos, sin, exp, pi
from numpy import random
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


scale_list = [0.97, 0.99, 1.01, 1.03]

#------------------------------ GENERATE POSCARs ----------------------------------
def prepPressureFit(top = '.', mustContain = 'test', scale_list = scale_list):
    """
    prepares POSCARs strained along a, b axes
    top: top-level directory (str)
    mustContain: name(s) for directory(ies) that must be included in root (str)
    scale_list: list of scaling factors along both lattice directions (list of floats)
    """
    # get paths to all POSCARs from previous test
    path_list = getPaths('POSCAR', top, mustContain)

    # get list of pressures to identify first level directory
    path_tab = [path.split('/') for path in path_list]
    pressure_list = [path_list[1] for path_list in path_tab]

    # create scaled POSCAR in directory labelled by scaling
    for path, pressure in zip(path_list, pressure_list):
        pos = p.POSCAR(path)
        for sa in scale_list:
            for sb in scale_list:
                scaledPos = pos.copy()
                directory = '%.3g_%.3g' %(sa, sb)
                scaledPos.cell[0] *= sa
                scaledPos.cell[1] *= sb
                destination = '%s/%s/%s' %(pressure, mustContain, directory)
                scaledPos.write('%s/POSCAR_scr' %destination)


#------------------------------ FIT PRESSURES ----------------------------------
def getPressures(
        top = '.',
        mustContain = 'test',
        ):
    """
    returns lists of lattice constants (Å) and corresponding pressure componenets (GPa)
    top: top-level directory (str)
    mustContain: name(s) for directory(ies) that must be included in root (str)
    """
    # get lattice constants from POSCARs
    a_list = []
    b_list = []
    path_list = getPaths('POSCAR', top, mustContain)    
    for path in path_list:
        with open(path) as f:
            for n in range(2):
                f.readline() # skip to cell
            a = norm([float(val) for val in f.readline().split()]) # Å
            b = norm([float(val) for val in f.readline().split()]) # Å

        a_list.append(a)
        b_list.append(b)

    # get pressure components from OUTCARs
    pa_list = []
    pb_list = []
    path_list = getPaths('OUTCAR', top, mustContain)    
    for path in path_list:
        with open(path) as f:
            for line in f:
                if 'in kB' in line:
                    line_list = line.split()
                    pa = float(line_list[2]) / 10 # GPa
                    pb = float(line_list[3]) / 10 # GPa

        pa_list.append(pa)
        pb_list.append(pb)

    return array(a_list), array(b_list), array(pa_list), array(pb_list)


def quadratic(c, a, b):
    """
    2D quadratic function
    c: six parameters (list of floats)
    lat: lattice constants (a, b) in Å (tuple of floats)
    """
    return c[0] + c[1]*a + c[2]*b + c[3]*a**2 + c[4]*a*b + c[5]*b**2


def quadraticErr(c, a, b, dataPt):
    """
    Error between quadratic fit and data
    c: parameters (list of floats)
    a, b: lattice constants (floats)
    dataPt: pressure component from data (float)
    """
    return quadratic(c, a, b) - dataPt

def getFit(
        a_ar = 'auto',
        b_ar = 'auto',    
        pa_ar = 'auto',
        pb_ar = 'auto',
        top = '.',
        mustContain = 'test',
        ):
    """
    fits polynomial function to pressure components to determine optimal a, b
    {a, b}_ar: lattice constants (a, b) for each strained system
        (array of tuples of floats)
        * if 'auto': grabs from POSCARs of prepPressureFit runs
    {pa, pb}_ar: pressure componenets (pa, pb) from each strained system
        (array of tuples of floats)
        * if 'auto': grabs from OUTCARs of prepPressureFit runs
    top: top-level directory (str)
    mustContain: name(s) for directory(ies) that must be included in root (str)
    """
    # get lattices constants and pressures
    if a_ar == 'auto':
        a_ar, b_ar, pa_ar, pb_ar = getPressures(top, mustContain)

    # quadratic fit: pressure vs. (a, b)
    guessParams = array([1, 1, 1, 1, 1, 1])
    R_list = []
    optParams_list = []
    for data in (pa_ar, pb_ar):
        optParams, success = leastsq(quadraticErr, guessParams[:],
                                     args = (a_ar, b_ar, data))
    
        # coefficients of determination
        mean = sum(data) / len(data)
        ss_tot = sum( [(pressure - mean)**2 for pressure in data] )
        ss_res = sum( (data - quadratic(optParams, a_ar, b_ar))**2 )
        R = 1 - ss_res / ss_tot
        print("coefficient of determination:\n    R = %.10g" %R)

        optParams_list.append(optParams)
        R_list.append(R)

    return optParams_list, R_list

def getEquations(latConstants, optParams_list):
    """
    returns pair of equations to be solved for a and b
    optParams_list: optimal parameters for pa vs. (a, b) returned from getFit()
        (list of 6 floats)
    """
    a, b = latConstants
    aParams, bParams = optParams_list
    return quadratic(aParams, a, b), quadratic(bParams, a, b)


def solve(
        a_ar = 'auto',
        b_ar = 'auto',
        pa_ar = 'auto',
        pb_ar = 'auto',
        top = '.',
        mustContain = 'test',
        ):
    """
    returns a, b, solutions to equation pa(a, b) and pb(a, b)
    """
    optParams_list, B_list = getFit(a_ar, b_ar, pa_ar, pb_ar, top, mustContain)
    return fsolve(getEquations, (4.3, 3.3), optParams_list)


def prepOptPOSCAR(
        a_ar = 'auto',
        b_ar = 'auto',
        pa_ar = 'auto',
        pb_ar = 'auto',
        top = '.',
        mustContain = 'test',
        write = True,
        printInfo = True,
        ):
    """
    creates POSCAR with lattice constants corresponding to pa = pb = 0
        * run in directory containing test_*
    """
    mustContain = [mustContain]
    posMustContain = mustContain  + ['1.00_1.00']
    
    # use coordinates from zero-pressure optimization
    path_list = getPaths('POSCAR_init', top, posMustContain)
    pos = p.POSCAR(path_list[-1])

    nexta, nextb = solve(a_ar, b_ar, pa_ar, pb_ar, top, mustContain)

    lasta = norm(pos.cell[0])
    lastb = norm(pos.cell[1])

    scalea = nexta / lasta
    scaleb = nextb / lastb

    pos.cell[0] *= scalea
    pos.cell[1] *= scaleb

    # write new POSCAR in directory test_${n} where n is next in sequence
    if write:
        dir_list = [d for d in os.listdir() if 'test' in d]
        dir_list.sort()
        lastDir = dir_list[-1]
        lastNum = int(lastDir[-2:])
        nextNum = lastNum + 1
        nextDir = 'test_%.2d' %nextNum
        outfile = '%s/1.00_1.00/POSCAR_init' %nextDir
        pos.write(outfile)

    return pos
    

def plotPressures(
        save = False,
        outfile = 'pressurePlot.pdf',
        top = '.',
        mustContain = 'test',
        ):
    """
    plots pressure vs. a, b data
    """
    a_ar, b_ar, pa_ar, pb_ar = getPressures(top, mustContain)

    # prepare figure and 3d axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(a_ar, b_ar, pa_ar)
    ax.scatter(a_ar, b_ar, pb_ar)

    plt.show()
    
def writePressures(outfile = 'data', top = '.', mustContain = 'test'):
    """
    writes pressures in readable format
    """
    a_ar, b_ar, pa_ar, pb_ar = getPressures(top, mustContain)
    with open(outfile, 'w') as f:
        for a, b, pa, pb in zip(a_ar, b_ar, pa_ar, pb_ar):
            f.write('%.8f,\t%.8f,\t%.8f,\t%.8f\n' %(a, b, pa, pb))
  
#------------------------------------- SCRATCH ----------------------------------------

# based on https://scipy-cookbook.readthedocs.io/items/FittingData.html

#fitfunc = lambda p, x: p[0]*cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function
#errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

def fitfunc(p, x):
    return p[0]*cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function

def errfunc(p, x, y):
    return fitfunc(p, x) - y # Distance to the target function

def genSinData(numPts = 150):
    """
    generates points with noise
    """
    t1 = linspace(5, 8, numPts)
    t2 = t1
    x1 = 11.86*cos(2*pi/0.81*t1-1.32) + 0.64*t1+4*((0.5-random.rand(numPts))*exp(2*random.rand(numPts)**2))
    x2 = -32.14*cos(2*pi/0.8*t2-1.94) + 0.15*t2+7*((0.5-random.rand(numPts))*exp(2*random.rand(numPts)**2))

    return t1, t2, x1, x2


def fitSin(numPts = 150):
    """
    fits 
    """
    t1, t2, x1, x2 = genSinData(numPts)

    p0 = [-15., 0.8, 0., -1.] # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(t1, x1))
    
    time = linspace(t1.min(), t1.max(), 100)
    plt.plot(t1, x1, "ro", time, fitfunc(p1, time), "r-") # Plot of the data and the fit
    plt.show()
    

#-------------------------- 2D Gaussian fit ---------------------------------
def gaussian(c, x, y):
    """Returns a gaussian function with the given parameters"""
    return c[0]*exp(
                -(((c[1]-x)/c[3])**2+((c[2]-y)/c[4])**2)/2)

def gaussianErr(c, x, y, dataPt):
    return gaussian(c, x, y) - dataPt

Xin, Yin = mgrid[0:201, 0:201]
x = ravel(Xin)
y = ravel(Yin)
data = ravel(gaussian([3, 100, 100, 20, 40], Xin, Yin) + random.random(Xin.shape))

def fitgaussian(data = data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    c_guess = array([3, 100, 100, 20, 40])
    p, success = leastsq(gaussianErr, c_guess[:], args = (x, y, data))
    return p

#------------------------------- Helpers ------------------------------------
def getPressure(infile = 'OUTCAR'):
    """ returns presure in kB from OUTCAR file """
    with open(infile) as f:
        for line in f:
            if 'pressure' in line:
                energy = float(line.split()[3])
    return energy

