# Anthony Yoshimura
# 10/07/17

from numpy import array, zeros, linspace, arange
from numpy import dot, tensordot, cross
from numpy import sqrt, sin, cos, arccos, radians, degrees, pi, floor, ceil, log10, exp
from numpy import random
from numpy.linalg import norm, inv
from subprocess import call 
from plotEnCon import getPaths, getEnergy
from POSCAR import POSCAR
#from matplotlib.mlab import griddata # deprecated
from scipy.interpolate import griddata
#import scipy as sc
import matplotlib.pyplot as plt
import os

top = '/Users/anthonyyoshimura/Desktop/koratkar/ReS2/neb'

#mustContain_list = ['climbA1toA1p/ph', 'climbA1ptoA1pp/ph', 'climbA2toA2p/ph', 'climbA2ptoA2pp/ph', 'climbA2pptoB/ph', 'climbA1pptoB/ph', 'climbA3toB/ph', 'climbA4toA4p/ph', 'climbA4ptoA1pp/ph', 'climbA3toA4/ph', 'climbA2pptoA3/ph', 'sdA1toA2/ph', 'miscEnergies', 'sdA4ptoA1/ph']

mustContain_list = [['cineb', 'ph'], ['landscape', 'ph'], 'miscEnergies']

#color_dict = {['A1toA1p', 'A1ptoA1pp']: 'blue',
#              ['A2toA2p', 'A2ptoA2pp', 'A2pptoB', 'BtoA1pp']: 'green',
#              ['A3toB', 'A1pptoB']: 'red',
#              ['A4toA4p', 'A4ptoA1pp']: 'orange'
#             }

color_list = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
bounds_list = [(0, 19), (19, 44), (44 , 58), (58, 70)]

def getNEBPaths(
        top = top,
        mustContain_list = mustContain_list,
        ):
    """
    returns paths leading to all POSCARs and OUTCARs in from neb runs
    top: top of directory tree (str)
    mustContain_list: list of strings that must be contained in path (list of str)
    """
    energyPaths = []
    poscarPaths = []
    for mustContain in mustContain_list:
        energyPaths += getPaths('OUTCAR', top, mustContain)
        poscarPaths += getPaths('POSCAR', top, mustContain)

    return poscarPaths, energyPaths 

def plot(
        atom = 17,
        ncontours = 30,
        save = False,
        outfile = 'landscape.pdf',
        points = 'neb',
        title = None):
    """
    plots energy landscape
    atom: atom as indexed in VESTA (pos int)
    """
    poscarPaths, energyPaths = getNEBPaths()
    globalPos = POSCAR(poscarPaths[0])
    globalPos.makeCartesian()
    globalCM = globalPos.getCM()

    # coords and energy lists for plotting
    x_list = []
    y_list = []
    energy_list = []

    for n in range(len(poscarPaths)):
        poscarPath = poscarPaths[n]
        energyPath = energyPaths[n]

        # align poscars to global CM
        pos = POSCAR(poscarPath)
        pos.makeCartesian()
        CM = pos.getCM()
        dif = globalCM - CM
        pos.translate(dif)

        # store x and y coords of atom
        x, y = pos.coords[atom - 1][:2]
        x_list.append(x)
        y_list.append(y)

        # store energy
        energy = getEnergy(energyPath)
        energy_list.append(energy)

    x_list = array(x_list) - min(x_list) + .1
    y_list = array(y_list) - min(y_list) + .1
    energy_list = array(energy_list) - min(energy_list)

    # define grid.
    xi = linspace(-.2, max(x_list) + .2, 100)
    yi = linspace(-.2, max(y_list) + .2, 100)

    # grid the data.
    zi = griddata(x_list, y_list, energy_list, xi, yi, interp='linear')

    # contour the gridded data, plotting dots at the nonuniform data points.
    CS = plt.contour(xi, yi, zi, ncontours, linewidths=0.5, colors='k')
    CS = plt.contourf(xi, yi, zi, ncontours, cmap = 'inferno',
                      vmax = zi.max(), vmin=zi.min())
    plt.colorbar()  # draw colorbar

    # all points or plot colored data point for each NEB path
    if points == 'neb':
        for n in range(len(color_list)):
            color = color_list[n]
            nmin, nmax = bounds_list[n]
            plt.scatter(x_list[nmin: nmax], y_list[nmin: nmax],
                        marker = 'o', s = 7, zorder = 10, color = color)
    else:
        plt.scatter(x_list, y_list, marker = 'o', s = 5, zorder = 10)

    plt.axes().set_aspect('equal')
#    plt.xlim(0, max(x_list) + .2)
#    plt.ylim(0, max(y_list))
    plt.xlim(0, 5)
    plt.ylim(0, 4)
#    plt.xticks(arange(0, max(x_list) + .2, 1))
#    plt.yticks(arange(0, max(y_list), 1))
    plt.xticks(arange(0, 6, 1))
    plt.yticks(arange(0, 5, 1))

    if type(title) == str:
        plt.title('Energy landscape of Re adsorption on ReS$_2$', fontsize = 14)
    plt.xlabel(r'x-position ({\AA})', fontsize = 14)
    plt.ylabel(r'y-position ({\AA})', fontsize = 14)
    
    plt.tight_layout()
    if save:
        plt.savefig(outfile)
    plt.show()

