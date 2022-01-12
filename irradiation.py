# Anthony Yoshimura
# 04/25/17

import POSCAR as p
from plotCross import getTmax
from numpy import array, dot, cos, sin, tan, arccos, arctan, sqrt, degrees, round, where
from periodic import table as ptable
from periodic import mp
import matplotlib.pyplot as plt

# lowest energy barrier is that of 00_10_00: ~0.297 eV

k  = 2.30707751e-28 # r^2/J
#m  = 9.10938356e-31 #kg
me = 5.109989461e5  #eV
#mp = 1.6726219e-27  #kg
mp = 9.3827231e8    #eV
Er = 2.17896e-18    #J
c  = 299792458      #m/s
a0 = 5.29177e-11    #m
ev = 1.60217662e-19 #C
e0 = 8.85418782e-12 #
ke = 2.30707751e-28 #r^2/J
kb = 8.6173303e-5   #eV/K
bm = 9.37e-11       #m  minimum impact parameter
alpha = 1/137
hbar = 6.582e-16    #eV s
#hbar = 1            #unitless


def getVandTheta(Tn, Te = 250000, species = 'Se'):
    """
    returns final velocity and angle of nucleus after collision
    Tn: kinetic energy transferred to nucleus in eV (pos float)
    Te: TEM energy in eV (pos float)
    species: atomic species of nucleus (str)
    """
    M = ptable[species][1] * mp 
#    root = mn*Tn*c**2 / (2*Te*(Te + 2*me*c**2))
    Tmax = getTmax(Te, M)
    root = sqrt(Tn / Tmax)
    vn = (2*Tn / M)**(1/2) * c / 100000   # A/fs
    return vn, arccos(root)

def getVandThetaFromB(b, Te = 30000, scatSpec = 'W', inSpec = 'Ga'):
    """
    returns final velocity and angle of nucleus after collision
    b: impact parameter in Angstrom (float)
    Te: Incident energy in eV (pos float)
    species: atomic species of nucleus (str)
    """
    M = ptable[species][1] * mp 
#    root = mn*Tn*c**2 / (2*Te*(Te + 2*me*c**2))
    Tmax = getTmax(Te, M)
    root = sqrt(Tn / Tmax)
    vn = (2*Tn / M)**(1/2) * c / 100000   # A/fs
    return vn, arccos(root)

def prepSputterSim(
        Tn = 7.0,
        TMD = 'WSe2',
        chal = True,
        dim = 5,
        infile = 'auto',
        outfile = 'POSCAR_scr',
        direction = [0, 0, 1]):
    """
    writes POSCAR with chalcogen initial velocity away from lattice
    Tn: initial kinetic energy in eV (float)
    TMD: chemical formula (str)
    chal: if True (False), simulates chalcogen (transition metal) sputtering.
    """
    if infile == 'auto':
        infile = '/Users/anthonyyoshimura/Desktop/meunier/trefoil/%s/unit/POSCAR' %TMD
    pos = p.POSCAR(infile)
    pos.makeSuperCell([dim, dim, 1])

    if chal:
        if 'Se2' in TMD:
            species = 'Se'
        elif 'S2' in TMD:
            species = 'S'
        atomNum = 51 # for trefoil project. POSCAR lists top Sulfur first
    else:
        if 'Mo' in TMD:
            species = 'Mo'
        elif 'W' in TMD:
            species = 'W'
        atomNum = 13

    M = ptable[species][1] * mp 
    vn = (2*Tn / M)**(1/2) * c / 100000   # A/fs

#    vector = dot([0, 0, vn], pos.cell_inv).tolist()
    pos.setIV([0, 0, vn], atomNum)
    pos.write(outfile)
    print('infile: %s' %infile)
    print('initial velocity: %s A/fs' %vn)
    print('outfile: %s' %outfile)
    print(pos.getSpecRanges())


def prepMigrationSim(
        Tn,
        phi = 0,
        Te = 80000,
        TMD = 'WSe2',
        dim = 5,
        outfile = 'POSCAR_scr',
        ):
    """
    writes POSCAR with chalcogen initial velocity
    phi: azimuth angle
    """
    infile = '/Users/anthonyyoshimura/Desktop/meunier/trefoil/%s/MD/defCon/\
              %sx%s/POSCAR' %(TMD,dim,dim)
    pos = p.POSCAR(infile)
    if 'Se2' in TMD:
        species = 'Se'
    elif 'S2' in TMD:
        species = 'S'

    vn, theta = getVandTheta(Tn, Te, species)
    c = vn * cos(theta)
    a = vn * sin(theta) * cos(phi)
    b = vn * sin(theta) * sin(phi)

#    vector = dot([a, b, c], pos.cell_inv).tolist()
    pos.setIV([a, b, c], 51) # chalgon is no. 51 for 5x5 cell
    pos.write(outfile)
    print('infile: %s' %infile)
    print('initial velocity: %s A/fs' %vn)
    print('theta: %s degrees' %degrees(theta))
    print('outfile: %s' %outfile)
    print(pos.getSpecRanges())

def prepImplantSim(Tn, phi = 0, Te = 80000, TMD = 'WSe2', dim = 5, outfile = 'POSCAR_scr'):
    """
    writes POSCAR with chalcogen initial velocity
    """
    infile = '/Users/anthonyyoshimura/Desktop/meunier/trefoil/%s/MD/defCon/%sx%s/POSCAR' %(TMD,dim,dim)
    pos = p.POSCAR(infile)
    if 'Se2' in TMD:
        species = 'Se'
    elif 'S2' in TMD:
        species = 'S'

    vn, theta = getVandTheta(Tn, Te, species)
    c = vn * cos(theta)
    a = vn * sin(theta) * cos(phi)
    b = vn * sin(theta) * sin(phi)

#    vector = dot([a, b, c], pos.cell_inv).tolist()
    pos.setIV([a, b, c], 51) # chalgon is no. 51 for 5x5 cell
    pos.write(outfile)
    print('infile: %s' %infile)
    print('initial velocity: %s A/fs' %vn)
    print('theta: %s degrees' %degrees(theta))
    print('outfile: %s' %outfile)
    print(pos.getSpecRanges())
#---------------------------------- PLOTTING --------------------------------------
def plotCompTMD(Tn = 7.5, atom = 51, show = True, save = False, outfile = 'sputTMDComp.pdf'):
    """
    plots z position (Angstrom) as a funciton of time (fs)
    atom: atom number as shown in VESTA (pos int)
    """
    fig, ax1 = plt.subplots()
#    ax2 = ax1.twinx()
    ax1.set_title('Chalgocen trajectories with %s eV' %Tn, fontsize = 14)
    ax1.set_xlabel('time (fs)', fontsize = 12)
    ax1.set_ylabel(r'distance from TMD surface ($\AA$)', fontsize = 12)
#    ax2.set_ylabel('acceleration ($\AA/fs^2$)', fontsize = 12)

    TMD_list = ['MoS2', 'WS2', 'MoSe2', 'WSe2']

    for TMD in TMD_list:
        infile = '/Users/anthonyyoshimura/Desktop/meunier/trefoil/%s/MD/sputter/%s/XDATCAR' %(TMD, Tn)
        with open(infile) as xdat:
            for n in range(4): xdat.readline()
            cell_height = float(xdat.readline().split()[-1])
            xdat.readline()
            pops = [int(val) for val in xdat.readline().split()]
            num_atoms = sum(pops)
            xdat_list = [line.split() for line in xdat.readlines()]
            coord_list = xdat_list[atom::num_atoms + 1]
            init_height = float(coord_list[0][2])
            z_ar = cell_height * (array([float(coord[2]) for coord in coord_list]) - init_height)
    
        domain = [n for n in range(len(z_ar))]
    
        ax1.plot(domain, z_ar, linewidth = 2, label = TMD)
#        ax2.plot(domain, z_ar - 1, linewidth = 2)

    diff_height = cell_height - init_height*cell_height
    ax1.text(.5, diff_height / (diff_height + 1), "top of cell", transform = ax1.transAxes, ha = 'center', va = 'bottom', fontsize = 12, color = 'blue')
    ax1.axhline(y = diff_height, ls = '--', color = 'blue')
    ax1.set_xlim([0, 300])
    ax1.set_ylim([0, diff_height + 1])

    ax1.legend()
    ax1.grid()

    plt.tight_layout()
    if save:
        plt.savefig(outfile)
    if show:
        plt.show()

def plotCompTn(
    TMD = 'MoS2',
    Tn_list = [7.5, 7.6, 7.7, 7.8, 7.9, 8.0],
    prefix_list = '',
    timeStep_list = 1,
    color_list = ['orange', 'blue', 'red', 'green'],
    atom = 51,
    figsize = (5, 6),
    title = True,
    show = True,
    save = False,
    outfile = 'sputTnComp.pdf'):
    """
    plots z position (Angstrom) as a funciton of time (fs)
    Tn_list: initial kinetic energy of chalcogen (list of pos floats)
    prefix: for special calculations e.g. 'vdw_', 't05_' (str)
    timeStep: time step in fs (pos float)
    atom: atom number as shown in VESTA (pos int)
    """
    # create figure
    fig, ax = plt.subplots(figsize = figsize)
    if title == True or title == 'auto':
        ax.set_title('%s trajectories for %s$_2$' %(TMD[:-2], TMD[:-1]), fontsize = 14)
    elif type(title) == str:
        ax.set_title(title, fontsize = 14)
    else:
        ax.text(.01, .99, '%s$_2$' %TMD[:-1], transform = ax.transAxes,
                fontsize = 14, ha = 'left', va = 'top')
    ax.set_xlabel('time (fs)', fontsize = 12)
    ax.set_ylabel(r'height ($\AA$)', fontsize = 12)
    domain_length = 0

    # make all parameters lists
    if type(timeStep_list) == float:
        timeStep_list = [timeStep_list for n in range(len(Tn_list))]
    if type(prefix_list) == str:
        prefix_list = [prefix_list for n in range(len(Tn_list))]

    # for loop over each directory
    for Tn, timeStep, prefix, color in zip(Tn_list, timeStep_list, prefix_list, color_list):

        # read XDATCAR from directory
#        infile = '/Users/anthonyyoshimura/Desktop/meunier/trefoil/%s/MD/sputter/%s%s/XDATCAR' %(TMD, prefix, Tn)
        infile = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/MD/sputter/%s/%s/XDATCAR' %(TMD, Tn)
        with open(infile) as xdat:
            for n in range(4):
                xdat.readline()
            cell_height = float(xdat.readline().split()[-1])
            xdat.readline()
            num_atoms = sum([int(val) for val in xdat.readline().split()])

            # extract z-position data for selected atom
            xdat_list = [line.split() for line in xdat.readlines()]
            coord_list = xdat_list[atom::num_atoms + 1]
            z_ar = array([float(coord[2]) for coord in coord_list])
            init_height = z_ar[0]

            # express z-position in cartestion coordinates relative to pristine position
            z_ar = cell_height * (z_ar - init_height + 1 - round(z_ar + .0000001))
            diff_height = cell_height - init_height*cell_height
            max_z = max(z_ar)

            # stop plot if atom turns around or passes the top of the cell
            if max_z < diff_height:
                max_index = where(z_ar == max_z)[0][0] 
            else:
                max_index = where(z_ar > diff_height)[0][0] 
            plus_list = z_ar[:max_index + 1]
                
        # time domain in fs
        domain = [n*timeStep for n in range(len(plus_list))]
        if len(domain) * timeStep > domain_length:
            domain_length = len(domain) * timeStep
        ls = '-'
        if timeStep != 1:
            ls = '--'
#        print(plus_list, z_ar)
        ax.plot(domain, plus_list, linewidth = 2, ls = ls,\
                color = color, label = '%s eV; %s fs' %(Tn, timeStep))

    # text, axis, and labels
    ax.text(.5, diff_height / (diff_height + 1), "top of cell",\
            transform = ax.transAxes, ha = 'center', va = 'bottom', fontsize = 12, color = 'blue')
    ax.axhline(y = diff_height, ls = '--', color = 'blue')
    ax.set_xlim([0, domain_length])
    ax.set_ylim([0, diff_height + 1])

    ax.legend(loc = 4, fontsize = 12)
    ax.grid()

    plt.tight_layout()
    if save:
        plt.savefig(TMD + outfile)
    if show:
        plt.show()

#---------------------------------- SCRATCH --------------------------------------
def getVandThetaformB(b, Te = 250000, atom = 'Se'):
    """
    returns classical chalcogen velocity and angle given the impact parameter
    b: impact parameter in meters (float)
    """
    mn = ptable[atom][1] * mp
    theta = arctan(2*b*Te / k)
    vn = (8*me*Te)**(1/2)/mn * cos(theta)

    return theta, vn

def get00_10_00(TMD = 'WSe2'):
    """
    returns 6x6 lattice with 00_10_00 vacancy configuration
    """
    pos = p.POSCAR('/Users/anthonyyoshimura/vaspFiles/POSCARs/POSCAR_%s' %TMD)
    pos.makeSuperCell([6,6,1])
    pos.remove([65, 66, 67, 79, 80, 81])
    return pos

def get11_11_11(TMD = 'WSe2', dim = 5):
    """
    returns TMD lattice with vacancy
    """
    pos = p.POSCAR('/Users/anthonyyoshimura/meunier/trefoil/%s/MD/isoMig/defCon/%sx%s/POSCAR' %(TMD,dim))
    return pos
