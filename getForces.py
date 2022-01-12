# Anthony Yoshimura
# 03/23/18

from numpy import sqrt, transpose
from numpy.linalg import norm
import matplotlib.pyplot as plt
from os import listdir

def getForces(
        infile = 'OUTCAR_929',
        atom_num = 49,
        ):
    """
    returns forces on specified atom throughout relaxation iterations
    infile: OUTCAR file (str)
    atom_num: atom number whose forces will be recorded (pos int)
    """
    # empty list to store forces
    force_list = []

    # read each line of OUTCAR
    with open(infile) as f:
        for line in f:

            if 'TOTAL' in line:
                # skip to atom's forces
                for n in range(atom_num):
                    f.readline()

                force = float(f.readline().split()[-1])
                force_list.append(force)

    return force_list
                    

def getRMSForce(
        infile = 'OUTCAR_929',
        ):
    """
    returns forces on specified atom throughout relaxation iterations
    infile: OUTCAR file (str)
    """
    # empty list to store forces
    rmsForce_list = []

    # read each line of OUTCAR
    with open(infile) as f:
        for line in f:

            if 'NION' in line:
                nions = int(line.split()[-1])

            if 'TOTAL' in line:
                # find rms force for each iteration
                squared_list = []
                f.readline()
                for n in range(nions):
                    force = float(f.readline().split()[-1])
                    squared_list.append(force**2)

                rmsForce = sqrt(sum(squared_list)) / nions
                rmsForce_list.append(rmsForce)

    return rmsForce_list


def getAvgForce(
        infile = 'OUTCAR_929',
        ):
    """
    returns forces on specified atom throughout relaxation iterations
    infile: OUTCAR file (str)
    """
    # empty list to store forces
    avgForce_list = []

    # read each line of OUTCAR
    with open(infile) as f:
        for line in f:

            if 'NION' in line:
                nions = int(line.split()[-1])

            if 'TOTAL' in line:
                # find avg force for each iteration (meV / A)
                force_list = []
                f.readline()
                for n in range(nions):
                    force_vector = [float(val) for val in f.readline().split()[-3:]]
                    force = norm(force_vector) * 1000 # meV / Å
                    force_list.append(force)
#                    force = float(f.readline().split()[-1])
#                    force_list.append(abs(force) * 1000)

                avgForce = sum(force_list) / nions
                avgForce_list.append(avgForce)

    return avgForce_list


def getMaxForce(
        infile = 'OUTCAR_929',
        ):
    """
    returns max forces (meV / Å) throughout relaxation iterations
    infile: OUTCAR file (str)
    """
    # empty list to store forces
    maxForce_list = []

    # read each line of OUTCAR
    with open(infile) as f:
        for line in f:

            if 'NION' in line:
                nions = int(line.split()[-1])

            if 'TOTAL' in line:
                # find max force for each iteration (meV / Å)
                force_list = []
                f.readline()
                for n in range(nions):
                    force_vector = [float(val) for val in f.readline().split()[-3:]]
                    force = norm(force_vector) * 1000 # meV / Å
                    force_list.append(force)

                maxForce = max(force_list)
                maxForce_list.append(maxForce)

    return maxForce_list


#--------------------------------- plotting ----------------------------------
def plot(
        infile = 'OUTCAR',
        system = 'BaMnSb$_2$',
        forceType = 'avg',
        save = False,
        outfile = 'ForceCon.pdf',
        ):
    """
    plots force as a function of ionic relaxation step
    infile: OUTCAR file (str)
    """
    # UNDER CONSTRUCTION: ignore atoms that selectively fixed
    fig, ax = plt.subplots()
    ax.set_title('%s ionic relaxation convergence' %system, fontsize = 15)
    ax.set_xlabel('ionic relaxation step', fontsize = 12)
  
    if forceType[0] == 'a' or forceType[0] == 'A':
        force_list = getAvgForce(infile)
        ax.set_ylabel(r'avg force (meV/$\AA$)', fontsize = 12)
    elif forceType[0] == 'm' or forceType[0] == 'M':
        force_list = getMaxForce(infile)
        ax.set_ylabel(r'max force (meV/$\AA$)', fontsize = 12)

    ax.plot(force_list)
    ax.set_ylim(0, max(force_list) * 1.1)
    ax.set_xlim(0, len(force_list))

    if save:
        plt.savefig('%s%s' %(forceType, outfile))

    plt.show()

#--------------------------------- From NEB ----------------------------------
def plotFromNEB(
        infile = 'OUTCAR',
        system = 'BaMnSb$_2$',
#        forceType = 'avg',
        save = False,
        outfile = 'nebForceCon.pdf',
        catch = False,
        ):
    """
    plots force as a function of ionic relaxation step for a set of NEB calculations
        * run with image files 00 01 02... in current directory
    infile: OUTCAR file (str)
    """
    # UNDER CONSTRUCTION

    # make list of directories containing OUTCARs
    dir_list = [f for f in listdir() if f.isdigit()]
    floatDir_list = sorted([float(val) for val in dir_list])
    imageDir_list = floatDir_list[1: -1]
    dir_list = ['%.2d' %f for f in imageDir_list]
#    dir_list = [f for f in os.listdir() if '_' in f]
#    allDir_list = [f for f in os.walk('.') if f.isdigit() and f not in dir_list]

    # gather max forces for each iteration
    imageForce_tab = []
    for d in dir_list:
        nebInfile = '%s/%s' %(d, infile)
        maxForce_list = getMaxForce(nebInfile)
        imageForce_tab.append(maxForce_list)

    # convert forces for each image to forces for each iteration
    iterForce_tab = transpose(imageForce_tab)
    maxForce_list = [max(force_list) for force_list in iterForce_tab]
    
    # plotting
    fig, ax = plt.subplots()
    ax.set_title('%s ionic relaxation convergence' %system, fontsize = 15)
    ax.set_xlabel('ionic relaxation step', fontsize = 12)
    ax.set_ylabel(r'max force (meV/$\AA$)', fontsize = 12)
    ax.plot(maxForce_list)
    ax.set_ylim(0, max(maxForce_list) * 1.1)
    ax.set_xlim(0, len(maxForce_list))

    if save:
        plt.savefig('%s' %outfile)
    plt.show()

    if catch:
        return maxForce_list
        
#--------------------------------- calling from bash ----------------------------------
if __name__ == '__main__':
    plot(system = '', save = True)
