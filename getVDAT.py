# Anthony Yoshimura
# 09/26/2017

from numpy import arange, array, dot
from makeReadable import alignColumns
import matplotlib.pyplot as plt
from periodic import table as p_dict
from numpy.linalg import norm
from getGCD import getGCD
from getChemForm import getChemForm
from os import getcwd

mp = 938272231 # eV
c = 299792458 # m/s

def xdat2vdat(infile = 'XDATCAR', outfile = 'VDAT', timeStep = 0.01, write = False, catch = True):
    """
    writes VDAT file containing velocities in A/fs based on XDATCAR data
    timeStep: time step (fs) used in MD calculations (float)
    """
    with open(infile) as xdat:
        xdat.readline()     # skip comment line
        scale = float(xdat.readline())
        cell = [ [float(val) for val in xdat.readline().split()] for n in range(3)]
        latLengths = [norm(vec) for vec in cell]
        specs = xdat.readline().split()
        pops = [int(val) for val in xdat.readline().split()]
        num_atoms = sum(pops)
    
        xdat_list = xdat.readlines()

    num_frames = int(len(xdat_list) / (num_atoms + 1))

    index_list = []
    coord_list = []
    for line in xdat_list:
        if 'configuration' in line:
            index_list.append(line)
        else:
            coord_list.append(line)

    coord_tab = [ [float(val) for val in line.split()] for line in coord_list]
    cart_tab = [dot(dir_coord, cell) for dir_coord in coord_tab]

    v_tab = []
    for n in range((num_frames - 2) * num_atoms):
            diff = cart_tab[n + 2 * num_atoms] - cart_tab[n]
            for n in range(3):
                diff[n] -= round(diff[n] / latLengths[n]) * latLengths[n]
            v = diff / timeStep / 2
            v_tab.append(v)

    # write to outfile
    if write:
    
        # align columns to improve readability
        v_tab_write = alignColumns(v_tab)
    
        with open(outfile, 'w') as f:
            f.write('Central difference velocities from %s assuming a time step of %s fs\n' %(infile, timeStep))
            for frame in range(num_frames - 2):
                f.write(index_list[frame + 1])
                for atom in range(num_atoms):
                    f.write('  %s\t' %(atom + 1))
                    for comp in range(3):
                        f.write(str(v_tab_write[frame * num_atoms + atom][comp]))
                        f.write('\t')
                    f.write('\n')
    
    # return table of velocities
    if catch:
        v_tab = array(v_tab)
        v_tab = v_tab.reshape(num_frames - 2, num_atoms, 3)
        return v_tab, num_frames - 2, num_atoms


#-------------------- ENERGETICS: needs OUTCAR, XDATCAR, and OSZICAR ---------------------
def getKinEn(
        atom_list,
        infile = 'XDATCAR',
        timeStep = 0.01,
        zero = False,
        zeroAtom = 'auto',
        ) :
    """
    returns kinetic energies for specified atoms throughout simulation
    atom_list: list of atoms as labelled in VESTA (list of pos ints)
    infile: XDATCAR file containing atomic positions (str)
    zero: if True, initial kinetic energy is set to zero (bool)
    zeroAtom: reference atom for initial velocity (pos int)
        * if 'auto': reference atom is the first atom in atom_list
    """
    # ensure atom_list is a list
    if type(atom_list) == int:
        atom_list = [atom_list]

    # get species and populations to determine masses
    with open(infile) as xdat:
        xdat.readline()     # skip comment line
        scale = float(xdat.readline())
        cell = [ [float(val) for val in xdat.readline().split()] for n in range(3)]
        specs = xdat.readline().split()
        pops = [int(val) for val in xdat.readline().split()]
        num_atoms = sum(pops)

    # create dictionary with atomic masses
    mass_dict = {}
    i = 0
    for spec, pop in zip(specs, pops):
        mass = p_dict[spec][1] * mp # mass in eV
        for n in range(pop):
            mass_dict[i] = (spec, mass)
            i += 1

    # get velocities for XDATCAR
    v_tab, num_frames, num_atoms = xdat2vdat(infile = infile, timeStep = timeStep)

    # set initial velocity to zero
    v_init = [0.0, 0.0, 0.0]
    if zero:
        if zeroAtom == 'auto':
            zeroAtom = atom_list[0]
        v_init = v_tab[0][zeroAtom - 1]
        print('subtracting velocity %s A/fs' %v_init)

    # collect kinetic energies for each atom
    ke_dict = {}
    for atom in atom_list:
        ke_list = [] # list of kinetic energies for a single atom
        mass = mass_dict[atom - 1][1]
        for frame in range(len(v_tab)):
            v = v_tab[frame][atom - 1] - v_init
            s = norm(v) * 1e5 / c # unitless speed
            ke = mass * s**2 / 2
            ke_list.append(ke)
        ke_dict[atom + 1] = ke_list 

    return ke_dict


def getTotEn(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        OUTCAR = 'OUTCAR',
        zero = True,
        zeroAtom = 'auto',
        outfile = 'totEn.pdf',
        save = False,
        ):
    """
    returns total energy (kinetic plus groundstate free) throughout simulation
    XDATCAR: XDATCAR file to determine velocities (str)
    OSZICAR: OSZICAR file to get GS free energies (str)
    OUTCAR: OUTCAR file to get time step (str)
    """
    # get time step (fs) and number of atoms from OUTCAR
    with open(OUTCAR) as f:
        for line in f:
            if 'NION' in line and '|   ' not in line: # ignore warning label
                numAtoms = int(line.split()[-1])
            elif 'POTIM' in line:
                timeStep = float(line.split()[2]) # fs

    # list of all atoms
    atom_list = [n + 1 for n in range(numAtoms)]

    # get kinetic energies (eV) from XDATCAR
    ke_dict = getKinEn(atom_list, infile = XDATCAR, timeStep = timeStep,
                       zero = True, zeroAtom = zeroAtom)

    # sum kinetic energies
    totKe_list = 0
    ke_tab = array([ke_dict[atom] for atom in ke_dict]).transpose()
    totKe_list = [sum(ke_list) for ke_list in ke_tab]

    # get free energies (eV) from OXZICAR
    F_list = []
    with open(OSZICAR) as f:
        for line in f:
            if 'F=' in line:
                F = float(line.split()[6])
                F_list.append(F)

    # add free and kinetic energies
    return [F + totKe for F, totKe in zip(F_list, totKe_list)], timeStep


#-------------------------------- PLOTTING ------------------------------------
def plotKinEn(
        atom_list,
        infile = 'XDATCAR',
        timeStep = 0.01,
        title = 'auto',
        save = False,
        outfile = 'auto',
        zero = False,
        zeroAtom = 'auto',
        ) :
    """
    plots kinetic energies for specified atoms throughout simulation
    atom_list: list of atoms as labelled in VESTA (list of pos ints)
    infile: XDATCAR file containing atomic positions (str)
    zero: if True, initial kinetic energy is set to zero (bool)
    zeroAtom: reference atom for initial velocity (pos int)
        * if 'auto': reference atom is the first atom in atom_list
    """
    # get kinetic energies from XDATCAR
    ke_dict = getKinEn(atom_list, infile, timeStep, zero, zeroAtom)

    # ensure atom_list is a list
    if type(atom_list) == int:
        atom_list = [atom_list]

    # get species and populations to determine masses
    with open(infile) as xdat:
        xdat.readline()     # skip comment line
        scale = float(xdat.readline())
        cell = [ [float(val) for val in xdat.readline().split()] for n in range(3)]
        specs = xdat.readline().split()
        pops = [int(val) for val in xdat.readline().split()]
        num_atoms = sum(pops)

    # plot
    fig, ax = plt.subplots()
    for atom in ke_dict:
        ke_list = ke_dict[atom]
        time_list = arange(len(ke_list)) * timeStep

        # label curve by impact parameter and azimuth angle
        # UNDER CONSTRUCTION: recognize from XDATCAR
        ax.plot(time_list, ke_list, lw = 2, label = 'atom %s' %(atom))

    # create title with chemical formula
    if title == 'auto':
       GCD = getGCD(pops[:-1])
       subscripts = [int(pop / GCD) for pop in pops[:-1]]
       chemFormula = ''
       for spec, subscript in zip(specs[:-1], subscripts):
           if subscript > 1:
               chemFormula += '%s$_%s$' %(spec, subscript)
           else:
               chemFormula += spec
       title = 'KE of %s through collision' %chemFormula

    ax.set_title(title, fontsize = 14)
    ax.set_ylabel('kinetic energy (eV)', fontsize = 12)
    ax.set_xlabel('time (fs)', fontsize = 12)
    ax.set_xlim(time_list[0], time_list[-1])
    ax.legend()
    ax.grid()
    if save:
        if outfile == 'auto':
            suffix = ''
            for atom in atom_list:
                suffix += '_%03d' %atom
            outfile = 'kinEn%s.pdf' %suffix
        plt.savefig(outfile)
    plt.show()


def plotTotEn(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        OUTCAR = 'OUTCAR',
        zero = True,
        zeroAtom = 'auto',
        outfile = 'totEn.pdf',
        save = False,
        tbounds = None,
        ebounds = None,
        note = 'auto',
        getVaspValues = False,
        ):
    """
    returns total energy (kinetic plus groundstate free) throughout simulation
    XDATCAR: XDATCAR file to determine velocities (str)
    OSZICAR: OSZICAR file to get GS free energies (str)
    OUTCAR: OUTCAR file to get time step (str)
    getVaspValues: if True, uses E from OSZICAR of NVE run (bool)
    """
    # get total energy list
    totEn_list, timeStep = getTotEn(XDATCAR, OSZICAR, OUTCAR, zero, zeroAtom)

    # get chemical formula from XDATCAR
    chemForm = getChemForm(XDATCAR)

    # time domain
    time_ar = arange(len(totEn_list)) * timeStep

    # create figure
    fig, ax = plt.subplots()
    ax.plot(time_ar, totEn_list, lw = 2, label = 'Ke + F')
    ax.set_title('Total energy of %s (%s-fs time step)' %(chemForm, timeStep), fontsize = 14)
    ax.set_xlabel('time (fs)', fontsize = 12)
    ax.set_ylabel('total energy (eV)', fontsize = 12)
    ax.grid()

    if tbounds == None:
        beg, end = time_ar[0], time_ar[-1]
    else:
        beg, end = tbounds
    ax.set_xlim(beg, end)

    if ebounds != None:
        top, bot = ebounds
        ax.set_ylim(top, bot)

    # add note in top right corner
    if note == 'auto':
        note = getcwd().split('/')[-1]
    ax.text(0.99, 0.99, note, fontsize = 12, verticalalignment = 'top',
            horizontalalignment = 'right', color = 'red', transform = ax.transAxes)

    # use E from OSZICAR
    E_list = []
    if getVaspValues:
        with open(OSZICAR) as f:
            for line in f:
                if 'E=' in line:
                    E = float(line.split()[4])
                    E_list.append(E)
        maxTimeIndex = len(time_ar)
        ax.plot(time_ar, E_list[:maxTimeIndex], lw = 2, label = 'OSZICAR')
        ax.legend(loc = 2)
        
    if save:
        plt.savefig(outfile)
    plt.show()
   

#-------------------------------------------------------------------------------
# calling from terminal
if __name__ == '__main__':
    xdat2vdat(write = True, catch = False)
