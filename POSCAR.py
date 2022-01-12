# Anthony Yoshimura
# 01/04/17

# POSCAR class for quick and intuitive manipulation of POSCARs in python
# Assumes atom species are given in line 6 of POSCAR

from numpy import array, arange, zeros, append
from numpy import dot, tensordot, cross, sign
from numpy import sqrt, sin, cos, arccos, radians, degrees, pi, floor, ceil,\
                  log10
from numpy.linalg import norm, inv

from subprocess import call
from inspect import stack
from copy import deepcopy
import os

from periodic import p_dict
from getGCD import getGCD
from getRREForm import getRREForm
from makeReadable import alignDecimals
import vasp.KPOINTS as KPOINTS

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

kb = 8.6173303e-5  # eV/K
me = 5.109989461e5  # mass of electron (eV)
mp = 9.3827231e8  # mass of proton (eV)
c  = 299792458  # speed of light (m/s)

#kb = 1.38064852e-23  # J/K
#mp = 1.6726219e-27  # kg 
#ev = 1.60217662e-19 # C


def getPOSCAR(infile='POSCAR'):
    """Returns a list of lists representing a POSCAR.

    infile: POSCAR file (str)
    """
    with open(infile) as pos_raw:
        pos_str_table = [n.split() for n in pos_raw]
    
        pos_table = pos_str_table[:]
        for x in range(len(pos_str_table)):
            for y in range(len(pos_str_table[x])):
                a = pos_str_table[x]
                try:
                    if a[y].isdigit():
                        b = int(a[y])
                    else:
                        b = float(a[y])
                    pos_table[x][y] = b 
                except ValueError:
                    pass

    return pos_table


def makePOSCAR(lattice=None, coords=0, lat_constant=2.0):
    """Returns POSCAR instance. """
    if lattice == None or lattice == 'sc':
        print('preparing empty cubic box')
        a = b = c = lat_constant
        d = 0

    elif lattice == 'fcc111':
        print('preparing empty hexagonal cell')
        a = lat_constant
        d = a / 2
        b = a * 3**(1/2) / 2
        c = a * 6**(1/2)       

    table = [[''],
             [1],
             [ a , 0.0, 0.0],
             [ d ,  b , 0.0],
             [0.0, 0.0,  c ],
             [],
             [],
             ['Direct'],
             []]

    return POSCAR(table)


def getFromXDAT(
        frame,
        infile = 'XDATCAR',
        timeStep = 'auto',
        OUTCAR = 'OUTCAR'):
    """Retuns POSCAR instance corresponding to specific frame in XDATCAR.

    infile: XDATCAR file (str)
    frame: frame or configuration number starting from 1 (pos int)
    timeStep: time step in fs used in MD.  Needed for velocities (float)
        * if 'auto', timeStep taken from POTIM in OUTCAR
    """
    # UNDER CONSTRUCTION: account for concatinated XDATCARs
    # time step from OUTCAR
    if timeStep == 'auto':
        with open(OUTCAR) as f:
            for line in f:
                if 'POTIM' in line:
                    timeStep = float(line.split()[2]) # fs
    
    # get POSCAR parameters from XDATCAR
    with open(infile) as f:
        system = f.readline().split()
        scale = float(f.readline())
        cell = [[float(val) for val in f.readline().split()]
                for n in range(3)]
        specs = f.readline().split()
        pops = [int(val) for val in f.readline().split()]
        num_atoms = sum(pops)

        # stop one frame before desired frame to get velocities
        # w/central difference
        while True:
            config_line = f.readline()
            if 'configuration' in config_line and str(frame-1) in config_line:
                break
            else:
                for n in range(num_atoms):
                    f.readline()

        # previous, current, and next coordinates
        coordsPrev = [[float(val) for val in f.readline().split()]\
                       for n in range(num_atoms)]
        f.readline()
        coords     = [[float(val) for val in f.readline().split()]\
                       for n in range(num_atoms)]
        f.readline()
        coordsNext = [[float(val) for val in f.readline().split()]\
                       for n in range(num_atoms)]

    # central difference for velocities (A/fs) in cartesian coordinates
    if timeStep > 1E-6:
        print('time step = %s fs' %timeStep)
        dirIvs  = ((array(coordsNext) - array(coordsPrev))\
                / timeStep / 2).tolist()
        cartIvs = [dot(dirIv, cell).tolist() for dirIv in dirIvs]
    else:
        cartIvs = []

    # construct POSCAR table
    pos_table = [system] + [[scale]] + cell + [specs, pops] + [['Direct']]\
              + coords + [[]] + cartIvs

    return POSCAR(pos_table)


def getFromQuanteumEspresso(infile='qscf.in', direct=True):
    """Retuns POSCAR instance from LAMMPS structure file.

    infile: LAMMPS structure file (str)
    """
    # UNDER CONSTRUCTION: accept more ATOM_POSITIONS and CELL_PARAMETERS types

    system = 'Structure from Quantum Espresso input file'
    atom_dict = {}
    cell = []
    coords = []

    with open(infile) as f:
        for line in f:
            if 'prefix' in line:
                system = line.split()[-1].strip("'")

            if 'CELL_PARAMETERS' in line:
                for n in range(3):
                    vec = [float(val) for val in f.readline().split()]
                    cell.append(vec)

            if 'ATOMIC_POSITIONS' in line:
                for line in f:
                    try:
                        spec, x, y, z, mx, my, mz = line.split()
                    except ValueError:
                        break

                    if spec in atom_dict:
                        atom_dict[spec] +=1
                    else:
                        atom_dict[spec] = 1

                    coords.append([float(val) for val in [x, y, z]])

    specs = list(atom_dict.keys())
    pops = list(atom_dict.values())

    pos_table = [[system]] + [[1.00]] + cell + [specs, pops] + [['Direct']]\
              + coords

    return POSCAR(pos_table)


def getFromLAMMPSTRJ(frame, infile):
    """Retuns POSCAR instance from LAMMPS structure file

    frame: frame or configuration number starting from 1 (pos int)
    infile: LAMMPS trajectory file (str)
    """
    # UNDER CONSTRUCTION: need to dumb for velocities

    system = 'Structure from lammpstr input file'
    atom_dict = {}
    cell = []
    coords = []

    with open(infile) as f:
        for line in f:
            if 'BOX BOUNDS' in line:
                for n in range(3):
                    vec = [0, 0, 0]
                    vec[n] = float(f.readline().split()[-1])
                    cell.append(vec)

            elif 'ATOMS id' in line:
                for line in f:
                    try:
                        n, spec, x, y, z = line.split()
                    except ValueError:
                        break

                    if spec in atom_dict:
                        atom_dict[spec] +=1
                    else:
                        atom_dict[spec] = 1

                    coords.append([float(val) for val in [x, y, z]])

    specs = list(atom_dict.keys())
    pops = list(atom_dict.values())

    pos_table = [[system]] + [[1.00]] + cell + [specs, pops] + [['Cartesian']]\
              + coords

    return POSCAR(pos_table)

def getFromLAMMPSDAT(infile, direct=True):
    """Retuns POSCAR instance from LAMMPS structure file.

    infile: LAMMPS structure file (str)
    direct: if True, POSCAR is written in direct coordinates (bool)
    """
    with open(infile) as f:

        system = f.readline().split()
        f.readline()
        numAtoms = int(f.readline().split()[0])
        numSpecs = int(f.readline().split()[0])
        f.readline()

        # cell
        norm_list = []
        for n in range(3):
            lo, hi = f.readline().split()[:2]
            norm = float(hi) - float(lo)
            norm_list.append(norm)
        
        xy, xz, yz = [float(val) for val in f.readline().split()[:3]]

        for n in range(3):
            f.readline()

        # get masses to determine species
        mass_list = []
        for n in range(numSpecs):
            mass = float(f.readline().split()[1])
            mass_list.append(mass)
        
        for n in range(3):
            f.readline()

        # coordinates
        coord_tab = [[] for n in range(numSpecs)]
        pops = [0 for n in range(numSpecs)]
        ID_dict = {}
        for n in range(numAtoms):
            ID, label, charge, x, y, z = f.readline().split()[:6]
            label = int(label) - 1
            ID_dict[ID] = label
            pops[label] += 1
            coord_tab[label].append([float(x), float(y), float(z)])
            
        for n in range(3):
            f.readline()

        # velocities
        iv_tab = [[] for n in range(numSpecs)]
        for n in range(numAtoms):
            ID, vx, vy ,vz = f.readline().split()
            label = ID_dict[ID]
            iv_tab[label].append([float(vx), float(vy), float(vz)])
            
    # prepare cell
    cell = zeros((3,3))
    cell[1][0] = xy
    cell[2][0] = xz
    cell[2][1] = yz
    cell[0][0] = norm_list[0]
    cell[1][1] = sqrt(norm_list[1]**2 - xy**2)
    cell[2][2] = sqrt(norm_list[2]**2 - xz**2 - yz**2)
    cell = cell.tolist()

    # get specs from masses
    specs = []
    for mass in mass_list:
        for spec in p_dict:
            if abs(p_dict[spec][1] - mass) < .1:
                specs.append(spec)
                break

    # prepare coords
    coords = []
    for coord_list in coord_tab:
        for coord in coord_list:
            coords.append(coord)

    if direct:
        for n in range(numAtoms):
            coords[n] = dot(coords[n], inv(cell)).tolist()

    # prepare velocities
    ivs = []
    for iv_list in iv_tab:
        for iv in iv_list:
            ivs.append(iv)

    rep = 'Cartesian'
    if direct:
        rep = 'Direct'

    pos_table = [system] + [[1.00]] + cell + [specs, pops] + [[rep]]\
              + coords + [[]] + ivs

    return POSCAR(pos_table)


#-------------------------- General Transformations ---------------------------
def genInvert(table, origin, atom_list):
    """
    returns a table in which the rows specified in atom_list are inverted
    table: list of cartesian coordinates (list of lists of floats)
    origin: point of inversion (list of 3 floats)
    atom_list: atoms, as listed in VESTA, to be inverted (list of ints)
    """
    atom_list = [atom - 1 for atom in atom_list] # python starts form 0

    # make sure we are not aliasing the old table
    new_table = array([coord[:] for coord in table])
    origin = array([origin])

    for atom in atom_list:
        new_table[atom] = 2*origin - table[atom] 

    return new_table.tolist()


def genReflect(table, rec_cell, plane, origin, atom_list):
    """
    returns table in which the rows specified in atom_list are reflected
    rec_cell: reciprocal lattice vectors (list of 3 lists of 3 floats)
    table: list of cartesian coordinates (list of lists of floats)
    plane: miller indices of reflection plane (list of 3 ints)
    origin: point of the reflection plane (list of 3 floats)
    """
    atom_list = [atom - 1 for atom in atom_list] # python starts from 0

    # make sure we are not aliasing the old table
    new_table = array([coord[:] for coord in table])
    origin = array(origin)

    # unit vector perpedicular to reflection plane
    perp = dot(array(plane), array(rec_cell))
    perp /= norm(perp)
     
    for atom in atom_list:
        # dist from origin projected onto perp
        dist = dot((new_table[atom] - origin), perp) * perp
        new_table[atom] -= dist * 2

    return new_table.tolist()


def genRotate(table, angle, axis, origin, atom_list):
    """
    returns table in which the rows specified in atom_list are rotated
    table: list of cartesion coordinates (list of lists of floats)
    angle: angle of rotation in degrees (float)
    axis: axis of rotation (list of 3 floats)
    origin: point on the axis (list of 3 floats)
    atom_list: atoms, as listed in VESTA, to be rotated (list of ints)
    """
    atom_list = [atom - 1 for atom in atom_list] # python starts from 0

    # make sure we are not aliasing the old table
    new_table = [coord[:] for coord in table]

    for atom in atom_list:
        vector = array(new_table[atom]) - array(origin)
        x = radians(angle) 
        u = array(axis) / norm(axis)

        sym = tensordot(u, u, 0)
        asym = array([[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]]) 
        I = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        R = cos(x)*I + sin(x)*asym + (1 - cos(x))*sym 

        # reverse mult order because matrices are transpose
        shift_vector = dot(vector, R)
        new_table[atom] = (shift_vector + origin).tolist()

    return new_table


def genTranslate(table, translation, atom_list):
    """
    returns table in which the rows specified in atom_list are translated 
    atom_list: atoms, as indexed in VESTA, to be moved (list of ints)  
            * if atom_list = 'all', translates all atoms
    translation: translation vector in direct coordinates (list of 3 floats)
    """
    atom_list = [atom - 1 for atom in atom_list] # python starts from 0

    # make sure we are not aliasing the old table
    new_table = [coord[:] for coord in table]

    for atom in atom_list:
        for vector in range(3):
            new_table[atom][vector] = new_table[atom][vector]\
                    + translation[vector]

    return new_table


#------------------------------- POSCAR Class ---------------------------------
class POSCAR(object):
    """
    Mutable representation of a POSCAR as a list of lists
    """
    # UNDER CONSTRUCTION: get ktable from instance of KPOINTS class

    def __init__(self, table = 'POSCAR', ktable = None):
        """
        Imports POSCAR (and KPOINTS) data from either a file or a table
        table: POSCAR file or table (str of list of lists)
        ktable: KPOINTS file, table, or instance
            (str, list of lists, or KPOINTS instance)
        """
        if type(table) == list:
            self.table = table
        elif type(table) == str:
            self.table = getPOSCAR(table)
        else:
            print('table must be a file or list of lists.')

        # add [:] to avoid aliasing
        self.name = self.table[:1][0][:]
        self.scale = self.table[1:2][0][:]
        self.cell = array([[float(val) for val in vec[:]]\
                            for vec in self.table[2:5]])
        self.cell_inv = inv(self.cell) # for switching cart <--> direct
        self.specs = self.table[5:6][0][:] # order matters. can't be a dict
        self.pops = self.table[6:7][0][:]
        if self.table[7:8][0][0][0] == 's' or self.table[7:8][0][0][0] == 'S':
            self.SD = True
            i = 1
        else:
            self.SD = False
            i = 0
        self.rep = self.table[7 + i: 8 + i][0][0]

        num_atoms = self.getNumAtoms()
        self.coords = [pos[:3] for pos in self.table[8 + i: 8 + i + num_atoms]]
        self.sd = [pos[3:6] for pos in self.table[8 + i: 8 + i + num_atoms]]
        self.iv = self.table[9 + i + num_atoms: 9 + i + num_atoms*2]
        if self.iv == []:
            print('adding velocity = [0.0, 0.0, 0.0] for all atoms')
            self.iv = [[0.0, 0.0, 0.0] for n in range(num_atoms)]

        # table for KPOINTS if necessary
        if ktable != None:
            if type(ktable) == list:
                self.ktable = ktable
            elif type(ktable) == str:
                self.ktable = getPOSCAR(ktable)
            elif type(ktable) == KPOINTS.KPOINTS:
                self.ktable = ktable.k_tab
            else:
                print('ktable must be a file or list of lists.')

            # add [:] to avoid aliasing
            self.kcomment = self.ktable[:1][0][:]
            self.knumber = self.ktable[1:2][0][0]
            self.ktype = self.ktable[2:3][0][0]
            self.kmesh = self.ktable[3:4][0][:]
            self.kshift = self.ktable[4:5][0][:]
        else:
            self.ktable = None

    def __add__(self, other):
        Sum = self.copy()
        Sum.addMolecule(other)
        return Sum

    def __str__(self):
        return self.getName()

    def write(self,
            outfile = 'POSCAR_scr',
            koutfile = 'KPOINTS_scr',
            writeKPOINTS = True,
            VESTA = False,
            OVITO = False,
            readable = False):
        """
        writes self.table to outfile
        outfile: file to which self.table will be written (str)
        VESTA: if True, outfile is immediately opened in VESTA (bool)
        """
        # UNDER CONSTRUCTION: ovito should show bonds 
        # write POSCAR
        if '/' in outfile:
            os.makedirs(os.path.dirname(outfile), exist_ok = True)

        # align decimals for columns containing floats
        cell_l2 = alignDecimals(self.cell)
        coords_l2 = alignDecimals(self.coords)

        coordsAndSd = [coords_l2[n] + self.sd[n]\
                       for n in range(self.getNumAtoms())]

        table = [self.name] + [self.scale] + cell_l2 + [self.specs]\
            + [self.pops] + [[self.rep]] + coordsAndSd + [[]] + self.iv

        if self.SD:
            table.insert(7, ['Selective Dynamics'])

        with open(outfile, "w") as f:
            for row in table:
                for element in row:
                    f.write(str(element) + ' ')
                f.write('\n')

        # write KPOINTS
        if koutfile != None and self.ktable != None and writeKPOINTS == True:
            if '/' in koutfile:
                os.makedirs(os.path.dirname(koutfile), exist_ok = True)

            table = [self.kcomment] + [[self.knumber]] + [[self.ktype]] \
                + [self.kmesh] + [self.kshift]

            with open(koutfile, "w") as f:
                for row in table:
                    for element in row:
                        f.write(str(element) + ' ')
                    f.write('\n')

        # open POSCAR in VESTA
        if VESTA:
            call("open -a VESTA " + outfile, shell = True)

        # open POSCAR in OVITO
        if OVITO:
            call("ovito {} &".format(outfile), shell = True)


    def add(self,
            species,
            coords,
            sd_list = None,
            cartesian = False,
            iv_list = None,
            ):
        """Adds atoms of a specified species to specified coordinates.

        species: atomic species (str)
        coords: table of postitions in direct coordinates
            (list of lists of floats)
            * One coordinate can be a list of floats (not list of lists)
            * To give cartesian coordinates to a direct POSCAR,
                set cartesian = True
        sd: selective dynamics of added atom (list of 'T's or 'F's)
            If sd = [], empty list is attached to each added atom
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # convert list into a list of lists
        if type(coords[0]) != list:
            coords = [coords]

        # ensure that coordinate elements are floats
        if type(coords[0][0]) == int:
            for n in range(len(coords)):
                coords[n] = [float(val) for val in coords[n]]

        if self.SD:
            if sd_list == None:
                sd_list = [['T', 'T', 'T'] for coord in coords]
        else:
            sd_list = [[] for coord in coords]

        if iv_list == None:
            iv_list = [[0.0, 0.0, 0.0] for coord in coords]

        # convert list into a list of lists
        if type(iv_list[0]) != list:
            iv_list = [iv_list]

        # ensure that coordinate elements are floats
        if type(iv_list[0][0]) == int:
            for n in range(len(iv_list)):
                iv_list[n] = [float(val) for val in iv_list[n]]

        if cartesian:
            coords = [self.getDirOf(coord) for coord in coords]

        if species in self.specs:
            ranges = self.getSpecRanges()
            for n in range(len(coords)):
                self.coords.insert(ranges[species][1], coords[n])
                self.sd.insert(ranges[species][1], sd_list[n])
                self.iv.insert(ranges[species][1], iv_list[n])

        else:
            self.coords += coords
            self.sd += sd_list
            self.iv += iv_list

        pop = len(coords)
        self.addToPops( {species: pop} )

    def addKPOINTS(self, ktable = None):
        """
        adds ktable
        ktable: KPOINTS file or table (str or list of lists)
        """
        if type(ktable) == list:
            self.ktable = ktable
        elif type(ktable) == str:
            self.ktable = getPOSCAR(ktable)
        elif ktable == None:
            print('Setting single k-point at gamma')
            self.ktable = [['automatic', 'mesh'], [0],
                           ['gamma'], [1, 1, 1], [0, 0, 0]]
        else:
            print('ktable must be a file or list of lists.')

        # add [:] to avoid aliasing
        self.kcomment = self.ktable[:1][0][:]
        self.knumber = self.ktable[1:2][0][0]
        self.ktype = self.ktable[2:3][0][0]
        self.kmesh = self.ktable[3:4][0][:]
        self.kshift = self.ktable[4:5][0][:]

    def addMolecule(self, other, origin = [0, 0, 0], otherOrigin = [0, 0, 0]):
        """
        adds the structure from another POSCAR
        other: instance of POSCAR class (POSCAR)
        origin: location in direct coordinates at which molecule
            will be inserted
            (with respect to molecule's origin aka otherOrigin)
            (list of 3 floats)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # convert site index to is a coordinate
        if type(origin) == int:
            origin = self.coords[origin - 1][:]

        # convert to cartesian coordinates to add positions
        mol = other.copy()
        mol.coords = (array(mol.coords) - array(otherOrigin)).tolist()
        if self.rep[0] == 'd' or self.rep[0] == 'D':
            mol.makeCartesian()

        # convert otherOrigin to cartesian
        if type(otherOrigin) == int:
            otherOrigin = mol.coords[otherOrigin - 1][:]
        otherOrigin = mol.getCartOf(otherOrigin)
        
        # other's initial velocities; already in cartesian, don't have to scale
        iv_list = [iv[:] for iv in mol.iv]

        # location of each species in mol.coords
        mol_ranges = mol.getSpecRanges()

        # add molecule's coordinates and species
        for spec in mol.specs:
            first, last = mol_ranges[spec]
            spec_coords = mol.coords[first: last]
            dir_coords = [(self.getDirOf(coord) + array(origin)).tolist()
                         for coord in spec_coords]
            spec_ivs = iv_list[first: last]
            self.add(spec, dir_coords, iv_list = spec_ivs) 

    def addToPops(self, atom_dict):
        """
        adds to element-populations in self.pops
        atom_dict: contains species and population of added atoms
            (dict {str: int})
            (dict {str: int})
        """
        for atom in atom_dict:
            if atom in self.specs:
                index = self.specs.index(atom) 
                self.pops[index] += atom_dict[atom] 
            else:
                self.specs.append(atom)
                self.pops.append(atom_dict[atom])

    def alignCell(self, zero=True, lat_num=1, direction=[1, 0, 0]):
        """Aligns cell so that a1 has no y, z components,
            and a2 has no z component.

        zero: if True, sets 2nd and 3rd compoenents of a1 and third components
            of a2 to exactly 0 (bool)
        lat_num: lattice vector number (1, 2, or 3)
        direction: direction (cartesian) to which lat_vector should align
            (list of 3 floats)
        """
        # UNDER CONSTRUCTION: Need to align with z AND x axes
        if stack()[1][3] == '<module>':
            self.updateTables()

        # aligning to z-axis
        # get vector perpendicular to atoms
        cell_perp = cross(self.cell[0], self.cell[1])

        # get vector perpendicular to plane
        plane_perp = [0, 0, 1]

        # get axis by which atoms can align to plane in one rotation
        axis = cross(cell_perp, plane_perp)

        # get angle between atomic and miller planes
        angle = self.getAngle(cell_perp, plane_perp, cartesian = True)
        if angle < 1e-6:
            print('Cell is already aligned')

        # rotate by appropriate angle along appropriate axis
        else:
            new_cell = genRotate(self.cell, angle, axis, [0, 0, 0], [1, 2, 3])
            self.cell = array(new_cell)
    
            # align a1 to [1, 0, 0]
            lat_vector = self.cell[0][:]
            direction = [1, 0, 0]
            axis = [0, 0, 1]
            angle = self.getAngle(direction, lat_vector, cartesian = True)

            # rotate in correct direciton, since getAngle only
            #     returns pos angles
            c = 1
            if lat_vector[1] > 0:
                c = -1
            new_cell = genRotate(self.cell, c*angle, axis,
                                 [0, 0, 0], [1, 2, 3])
            self.cell = array(new_cell)

            if zero:
                self.cell[0][1] = self.cell[0][2] = self.cell[1][2] = 0.0
    
    def alignToGrid(self,
            dim = [3, 3, None],
            atom_list = 'all',
            origin = [0, 0, 0],
            thres = 0.1, # Å
            ):
        """
        aligns atom coordinates to nearest grid point in direct coordinates
            * Grids run parallel to lattice vectors
        dim: grid dimensions (list of 3 pos ints)
            * if 'None', no alignment in that direction
        origin: origin of grid in direct coordinates (list of floats)
            * if int, origin is coordinates of corresponding atom
        thres: if distance between coordinate componenet and gridplane in Å
            is greater than thres, compoenent will not be rewritten (float)
        """
        # UNDER CONSTRUCTION: still needs debugging.  May replace atoms
        #     that are beyond the theshold.
        if stack()[1][3] == '<module>':
            self.updateTables()

        # prepare list of atoms that will be aligned
        if atom_list == 'all':
            atom_list = [atom for atom in range(self.getNumAtoms())]
        else:
            atom_list = [atom - 1 for atom in atom_list]

        # set origin on selected atom
        if type(origin) == int:
            origin = self.coords[origin - 1][:]
 
        # list of dimensions that can be iterated through
        trueDim = []
        for comp in dim:
            if comp == None:
                trueDim.append(1)
            else:
                trueDim.append(comp)

        # prepare list of lists of grid-plane locations in each direction
        grids = [[point / trueDim[comp] + origin[comp]\
                for point in range(trueDim[comp])] for comp in range(3)]

        # set each coordinate component to closest grid plane
        newCoords = []

        # take transpose of coordinates so that if statement only comes up
        #     once per component
#        transCoords = transpose(self.coords)
        for atom, coord in enumerate(self.coords):
            newCoord = []

            for vec, comp, grid, dimComp in zip(list(range(3)),
                                                coord, grids, dim):
                if dimComp == None:
                    newComp = comp

                # find closest grid-plane to coordinate component
                else:
                    minDist = 1
                    newComp = comp
                    for point in grid:
                        dist = comp - point
                        dist -= round(dist) # only consider nearest image
                        dist = abs(dist)
                        if dist < minDist:
                            minDist = dist
                            newComp = point

                    # don't replace component if it is too far from grid plane
                    cartMinDist = norm(self.cell[vec] * minDist)
                    if cartMinDist > thres:
                        newComp = comp
                        print('component %s of atom %s was not within %s Å of\
                               a grid plane.'
                               %(vec, atom, thres))

                newCoord.append(newComp)
            newCoords.append(newCoord)
                
        # update coordinates
        self.coords = newCoords
                
    def alignToLattice(self, atom1, atom2, vector = 'auto'):
        """
        aligns vector connecting atom1 and atom2 to a lattice vector
        atom1 and atom2: pair of atoms to be aligned (pos ints)
        vector: lattice vector to which atoms will be aligned (1, 2, or 3)
            if 'auto', aligns to vector that is already most closely aligned
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        atom_vector = self.getCartOf(self.getAxis(atom1, atom2))

        # find most nearly aligned lattice vector
        if vector == 'auto':
            vector = 2
            best_alignment = abs(dot(atom_vector, self.cell[2]))
            for n in range(2):
                unit_vector = self.cell[n] / norm(self.cell[n])
                alignment = abs(dot(atom_vector, unit_vector))
                if alignment > best_alignment:
                    vector = n
        else:
            vector -= 1

        # rotate by appropriate angle along appropriate axis
        lat_vector = self.cell[vector]
        angle = self.getAngle(lat_vector, atom_vector, cartesian=True,
                              deg=True)
        axis = self.getDirOf(cross(atom_vector, lat_vector))
            # ^ perp to lat and atom vecssian = True)
        print('atom_vector =', atom_vector)
        print('lat_vector =', lat_vector)
        print('axis of rotation =', axis)
        print('angle =', angle)
        self.rotate(angle, axis, atom1)
        
    def alignToPlane(self, atom1, atom2, atom3, plane = (0, 0, 1)):
        """
        aligns plane spanned by atom1, atom2, and atom3 to a lattice plane
        atom{1,2,3}: atoms than span the plane (pos ints)
        plane: miller indices of the plane to which atoms are aligned
            (tup of 3 ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # get vector perpendicular to atoms
        atom_vector1 = self.getCartOf(self.getAxis(atom1, atom2))
        atom_vector2 = self.getCartOf(self.getAxis(atom1, atom3))
        atom_perp = cross(atom_vector1, atom_vector2)

        # get vector perpendicular to plane
        rec_cell = self.getRecip() 
        plane_perp = dot(plane, rec_cell)

        # get axis by which atoms can align to plane in one rotation
        axis = cross(atom_perp, plane_perp)
        axis = self.getDirOf(axis)

        # get angle between atomic and miller planes
        atom_perp = self.getDirOf(atom_perp)
        plane_perp = self.getDirOf(plane_perp)
        angle = self.getAngle(atom_perp, plane_perp)

        # rotate by appropriate angle along appropriate axis
        print('axis of rotation =', axis)
        print('angle =', angle)
        self.rotate(angle, axis, atom1)

    def bend(self, focus):
        """ Bends solid according to Euler-Bernoulli approximation """
        pass

    def bringIntoCell(self):
        """ Brings atoms outside the cell into cell """
        if stack()[1][3] == '<module>':
            self.updateTables()

        for i in range(len(self.coords)):
            for j in range(3):
                self.coords[i][j] -= floor(self.coords[i][j])

    def center(self, center_list='all', atom_list='all'):
        """Brings structure's center of mass to the center of the cell.

        center_list: list of axis along which to center (list of ints)
            * if 'all', centers along all three axes
        atom_list: atoms (labelled as in VESTA) that make up the structure
            (list of pos ints)
        """
        # UNDER CONSTRUCTION: only center along specified axes
        if stack()[1][3] == '<module>':
            self.updateTables()

        cm = self.getCM(atom_list)
      
        if self.rep[0] == 'C' or self.rep[0] == 'c':
            shift = (self.cell[0] + self.cell[1] + self.cell[2]) / 2 - cm
        elif self.rep[0] == 'D' or self.rep[0] == 'd':
            shift = array([.5, .5, .5]) - cm

        self.translate(shift, atom_list)
            
    def changeCell(self, new_cell, origin=[0.0, 0.0, 0.0], strip=False):
        """Changes shape of cell while maintaining atom coordinates.

        new_cell: new cell (3x3 array)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        new_cell = [[float(val) for val in row] for row in new_cell]

        self.translate(-array(origin))
        self.makeCartesian()
        self.cell = array(new_cell)
        self.cell_inv = inv(self.cell)
        self.makeDirect()
        if strip:
            self.strip()

    def copy(self):
        """ returns identical POSCAR class """
        coordsAndSd = [self.coords[n] + self.sd[n]\
                       for n in range(self.getNumAtoms())]

        table = [self.name] + [self.scale] + self.cell.tolist() + [self.specs]\
              + [self.pops] + [[self.rep]] + coordsAndSd + [[]] + self.iv

        if self.SD:
            table.insert(7, ['Selective Dynamics'])

        if self.ktable != None:
            ktable = [self.kcomment] + [[self.knumber]] + [[self.ktype]] \
                   + [self.kmesh] + [self.kshift]
        else:
            ktable = None

        return POSCAR(table, ktable)

    def dope(self, species, atom_list):
        """
        Updates the species of the atoms in atom_list
        species: atomic species (str)
        atom_list: list of atomic sites, as indexed in VESTA (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(atom_list) == int:
            atom_list = [atom_list]

        insert_coords = [self.coords[atom - 1] for atom in atom_list]
        self.remove(atom_list)
        self.add(species, insert_coords)

    def getAngle(self, v1, v2, cartesian = False, deg = True):
        """
        returns the angle (in degrees) between two vectors 
        v1 and v2: vectors in direct coordinates (list or array of 3 floats)
        vertex: vertex atom (pos int)
        cartesian: set to True if self.rep is Cartesian
        deg: if True, angle given in degrees (bool)
            * if False, angle given in radians
        """
        if not cartesian:
            v1 = self.getCartOf(v1)
            v2 = self.getCartOf(v2)

        u1 = array(v1) / norm(v1)
        u2 = array(v2) / norm(v2)

        cosine = dot(u1, u2)
        angle = arccos(cosine)
 
        if deg:
            angle = degrees(angle)

        return angle

    def getAxis(self, atom1, atom2):
        """
        returns axis that connects two atoms in direct coordinates
        atoms: atoms indexed as in VESTA (pos int)
        """
        return array(self.coords[atom2 - 1]) - array(self.coords[atom1 - 1])

    def getBondAngle(self, end1, mid, end2, deg = True):
        """
        returns the bond angle in degrees between the two end points and
            a vertex
        ends: end point atoms (pos ints)
        vertex: vertex atom (pos int)
        """
        axis1 = self.getAxis(end1, mid)
        axis2 = self.getAxis(end2, mid)

        return self.getAngle(axis1, axis2, deg = deg)

    def getCartOf(self, site):
        """
        returns the cartesian coordinates of a particular site or vector
        site: site as indexed in VESTA or coordinate vector
            (pos int or list of 3 floats)
        """
        if type(site) == int:
            if self.rep[0] == 'c' or self.rep[0] == 'C':
                return self.coords[site - 1]
            else:
                site = array(self.coords[site - 1][:])

        return dot(site, self.cell).tolist()

    def getCellVolume(self):
        """ returns cell volume in cubic Angstroms """
        cross_product = cross(self.cell[0], self.cell[1])
        return abs( dot(self.cell[2], cross_product) )
        
    def getCM(self, atom_list = 'all'):
        """
        returns center of mass of atoms in atom_list
        atom_list: atoms as indexed in VESTA (list of pos ints)
        """
        if atom_list == 'all':
            atom_list = [n for n in range(self.getNumAtoms())]
        else:
            atom_list = [n - 1 for n in atom_list]

        tot_mass = 0
        cm = zeros(3)
        spec_dict = p_dict

        for atom in atom_list:

            spec = self.getSpecOf(atom + 1)
            mass = spec_dict[spec][1]
            tot_mass += mass
            cm += array(self.coords[atom]) * mass

        return cm / tot_mass

    def getCoordOf(self, atom):
        """
        returns direct coordinates for a given atom
        atom: atom index starting from 1 (pos int)
        """
        return array(self.coords[atom - 1])

    def getDirOf(self, site, sourceRep = 'cart', deg = True):
        """
        returns the direct coordinates of a particular site or vector
        site: site as indexed in VESTA or coordinate vector
            (pos int or list of 3 floats)
        sourceRep: 'cartesian' or 'spherical' (str)
            * spherical coordinates = (r, theta, phi)
            * theta and phi in degrees
        deg: if True, theta and phi are given in degrees
        """
        # convert atomic site number to coordinates
        if type(site) == int:
            site = site.coords[site - 1]
            sourceRep = self.rep
            
            # simply take coordinate from self.coords if rep is 'Direct'
            if sourceRep[0] == 'd' or sourceRep[0] == 'D':
                return site
       
        # convert spherical to cartesian
        if sourceRep[0] == 's' or sourceRep[0] == 'S':
            r, theta, phi = site
            if deg:
                theta, phi = radians(theta), radians(phi)
            x = r * sin(theta) * cos(phi)
            y = r * sin(theta) * sin(phi)
            z = r * cos(theta)
            site = array([x, y, z])

        # convert cartisian to direct
        return dot(site, self.cell_inv).tolist()

    def getDist(self, atom1, atom2):
        """
        returns distance between two atoms in Å
        atoms: atom index starting from 1 (pos int)
        """
        if self.rep[0] == 'c' or self.rep[0] == 'C':
            axis = self.getAxis(atom1, atom2)
            return norm(axis)

        else:
            x1C = array(self.getCartOf(atom1))
            x2C = array(self.getCartOf(atom2))
            return norm(x1C - x2C)

    def getFinMigPOSCAR(self,
            transform = 'translation',
            translation = [.5, 0, 0],
            origin = [.5, .5, .5],
            axis = [0, 0, 1],
            angle = 0,
            threshold = 1,
            ):
        """
        returns POSCAR instance for system after migration
            * takes initial system and translates in the desired direction,
               renordering the coordinates to best match the initial self 
            * self should be a fully relaxed initial system.
            * inital and final systems should be identical besides a
                global translation
        transform: type of symmetry transform
            (translation, rotation, reflection, inversion) (str)
        translation: vector (direct coords) in which system tranlates
            (list of 3 floats)
        origin: symmetry point for invsersions, reflections, and rotations
            (list of 3 floats)
        axis: symmetry vector for reflections and rotations (list of 3 floats)
            * reflections: miller indices of plane, rotations: axis of rotation
        angle: angle of rotation in degrees (float)
        threshold: max dist in Å for which an atom is recognized as
            the same atom (float)
        """
        # final positions at the end of migration
        fin = self.copy()

        # origin on an atom's center
        if type(origin) == int:
            origin = self.coords[origin - 1]
        
        # translation
        if transform[0] == 't' or transform[0] == 'T':
            fin.translate(translation)

        # rotation
        elif transform[:3] == 'ro' or transform[:3] == 'Ro'\
            or transform[:3] == 'RO':
            fin.rotate(angle, axis, origin)

        # reflection
        elif transform[:3] == 're' or transform[:3] == 'Re'\
            or transform[:3] == 'RE':
            fin.reflect(axis, origin)

        # inversion
        elif transform[0] == 'i' or transform[0] == 'I':
            fin.invert(origin)

        # glide
        elif transform[0] == 'g' or transform[0] == 'G':
            fin.reflect(axis, origin)
            fin.translate(translation)

        fin.bringIntoCell()

        # empty POSCAR in which final positions will be added in correct order
        new = fin.copy()
        finAtoms = [n + 1 for n in range(self.getNumAtoms())] # coord indices
        new.remove(finAtoms)

        # make copy of coordinate lists (mostly for legibility)
        initCoords = deepcopy(self.coords)
        finCoords = deepcopy(fin.coords)

        # find nearest atoms between initial and final POSCARs
        for initAtom, initCoord in enumerate(self.coords):
            initCoord = array(initCoord)
            minDist = norm(self.cell) # upper bound for interatomic distance
            minCoord = None
            minAtom = None

            # find atom closest to initial coord
            for finAtom, finCoord in zip(finAtoms, finCoords):
                finCoord = array(finCoord)
                difCoord = initCoord - finCoord

                # Only consider nearest image
                for n in range(3):
                    difCoord[n] -= round(difCoord[n])
                    
                # dif becomes new minDist if it is smaller
                dif = norm(self.getCartOf(difCoord))
                if dif < minDist:
                    minDist = dif
                    minCoord = finCoord
                    minAtom = finAtom

            # identify the atom that migrated
            if minDist > threshold:
                migAtom = initAtom + 1
                print('migrating atom: %s' %migAtom)

            # add coordinates to new POSCAR and remove coordinates from list
            else:
                finAtoms.remove(minAtom)
                newCoord = minCoord.tolist()
                finCoords.remove(newCoord)
                newSpec = fin.getSpecOf(minAtom) # species from fin
                new.add(newSpec, newCoord)

        # insert migrating atom in correct position
        migSpec = self.getSpecOf(migAtom)
        new.add(migSpec, finCoords[0])
        lastAtom = new.getSpecRanges()[migSpec][1]
        new.coords.insert(migAtom - 1, new.coords.pop(lastAtom - 1))

        return new

    def getK_list(self, kmin = 3, kmax = 12, layer = False):
        """
        returns list of k-point mesh dimensions suitable for cell
        kmin: min number of kpoints along shortest axis (pos int)
        kmax: max number of kpoints along shortest axis (pos int)
        layer: if True, third mesh dimenstion = 1 (bool)
        """
        length_list = [norm(vec) for vec in self.cell]
        shortestLength = min(length_list)

        k_list = []
        for n in range(kmin, kmax + 1):
            mesh = []
            for comp, length in enumerate(length_list):
                dim = int(round((n + 1) * shortestLength / length))
                if dim == 0:
                    dim = 1
                if comp == 2 and layer:
                    dim = 1
                mesh.append(dim)
            k_list.append(mesh)

        return k_list                

    def getKinEnOf(self, atom):
        """
        returns the kinetic energy in eV of an atom based on its velocity
        atom: site as intexed in VESTA (pos int)
        """
        # UNDER CONSTRUCTION: make compatible with mp in eV
        spec = self.getSpecOf(atom)
        m = p_dict[spec][1] * mp #kg
        v = self.iv[atom - 1] # Å/fs
        vSquared = dot(v, v) * 1e10 #m^2/s^2
        return m * vSquared / 2 / ev

    def getLatConstants(self):
        """
        returns the lattice constants a, b, c
        """
        a, b, c = [norm(vector) for vector in self.cell]
        return a, b, c

    def getLayers(self, column = 2, max_thickness = 0.6):
        """
        returns dict with sites (VESTA) and layer number 
            {layer: [(height, site number, species), (), ...]}
        column: lattice vector that intercects layers
        """
        # sort sites by increasing height
        height_list = [(self.coords[site][column], site + 1,
                        self.getSpecOf(site + 1))
                       for site in range(self.getNumAtoms())]
        height_list.sort()

        # group sites whose heights are within 0.1 Angstrom of eachother
        scale = 1
        if self.rep[0] == 'd' or self.rep[0] == 'D':
            scale *= norm(self.cell[column])
        layer = 1
        height = height_list[0][0]
        layer_dict = {layer: [height_list[0]]}

        for height_tup in height_list[1:]:
            dif = (height_tup[0] - height) * scale
            if dif < max_thickness:
                layer_dict[layer].append(height_tup)
            else:
                layer += 1
                height = height_tup[0]
                layer_dict[layer] = [height_tup]

        return layer_dict 

    def getName(self, LaTeX = True):
        """ returns the chemical formula of the system """
        GCD = getGCD(self.pops)
        subscripts = [int(pop / GCD) for pop in self.pops]
        chemFormula = ''
        for spec, subscript in zip(self.specs, subscripts):
            if subscript > 1:
                if LaTeX:
                    chemFormula += '%s$_{%s}$' %(spec, subscript)
                else:
                    chemFormula += '%s%s' %(spec, subscript)
            else:
                chemFormula += spec

        return chemFormula

    def getNN(self, center):
        """
        returns dict with sites (VESTA) and distances of a center's
            nearests neighbors
        center: atomic site as labelled in VESTA (pos int)
        """
        maxIndex = self.getNumAtoms() + 1

        # sort sites by distance to center
        dist_list = [(self.getDist(center, site), site, self.getSpecOf(site))
                        for site in range(1, maxIndex) if site != center]
        dist_list.sort()

        # group sites whose distance from the center are nearly identical
        degree = 1
        nn_dist = dist_list[0][0]
        nn_dict = {degree: [dist_list[0]]}

        for dist_tup in dist_list[1:]:
            if dist_tup[0] - nn_dist < 0.01:
                nn_dict[degree].append(dist_tup)
            else:
                degree += 1
                nn_dist = dist_tup[0]
                nn_dict[degree] = [dist_tup]

        return nn_dict

    def getNumAtoms(self):
        """ returns the number of atoms in the system """
        return sum(array(self.pops))

    def getPerp(self, atom1, atom2, atom3):
        """
        returns cartesian unit vector perpendicular to plane spanned by atoms
        atom1-3: atomic site as labelled in VESTA (pos ints)
        """
        # get vector perpendicular to atoms
        atom_vector1 = self.getCartOf(self.getAxis(atom1, atom2))
        atom_vector2 = self.getCartOf(self.getAxis(atom1, atom3))
        atom_perp = cross(atom_vector1, atom_vector2)
        atom_perp /= norm(atom_perp)

        return atom_perp

    def getPerpDist(self, atom, atom1, atom2, atom3):
        """
        returns distance between atom and plane spanned by atoms 1-3
        atom: atomic site as labelled in VESTA (pos int)
        atom1-3: atomic sites of atoms that span plane (pos ints)
        """
        perp = self.getPerp(atom1, atom2, atom3)
        axis = self.getCartOf(self.getAxis(atom, atom1))
        perp_dist = dot(perp, axis)

        return perp_dist

    def getRecip(self):
        """ returns reciprical cell as a 2D array """
        rec_cell = zeros([3, 3])
        
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            cross_product = cross(self.cell[j], self.cell[k])
            volume = dot(self.cell[i], cross_product)
            rec_cell[i] = 2 * pi * cross_product / volume

        return rec_cell

    def getSpecOf(self, index):

        """ returns species of atom with given index (VESTA) """
        range_dict = self.getSpecRanges()
        for spec in range_dict:
            lower, upper = range_dict[spec]
            if index > lower and index <= upper:
                return spec

        print('index out of range')

    def getSpecRanges(self):
        """
        returns dictionary of specs and their indices ranges as shown in Vesta
            {spec: [first, last]}
        """
        spec_ranges = {}
        index = 0
        for n in range(len(self.specs)):
            spec_ranges[self.specs[n]] = (index, index + self.pops[n])
            index += self.pops[n]

        return spec_ranges

    def getSphOf(self, site, sourceRep = 'dir', deg = True):
        """
        returns the spherical coordinates of a particular site of vector
            * r in Å, theta and phi in degrees
        site: site indexed as in VESTA or coordinate vector
            (pos int or list of 3 floats)
        sourceRep: 'direct' or 'cartesian' (str)
        deg: if True, theta and phi are given in degrees
        """ 
        # convert atomic site number to coordinates
        if type(site) == int:
            site = self.coords[site - 1]
            sourceRep = self.rep

        # convert direct coordinates to cartesian
        if sourceRep[0] == 'd' or sourceRep[0] == 'D':
            site = dot(site, self.cell)

        # calculate r, theta, and phi
        r = norm(site)
        x, y, z = site[0], site[1], site[2]
        theta = arccos(z / r)
        try:
            phi = arccos(x / sqrt(x**2 + y**2))
        except RuntimeWarning:
            print('theta = 0, arbitrarily setting phi = 0')
            phi = 0.0

        # convert to degrees
        if deg:
            theta, phi = degrees(theta), degrees(phi)
        
        return r, theta, phi

    def getTemp(self):
        """
        returns the tempurature associated with the initial velocities
        """
        # get the masses for all atoms in POSCARs
        specRanges_dict = self.getSpecRanges()
        m_list = []
        for spec in specRanges_dict:
            m = p_dict[spec][1] * mp
            min, max = specRanges_dict[spec]
            for n in range(min, max):
                m_list.append(m)
                
        # get the displacement between POSCARs 0 and 1
        const = 1e10 # (A/fs)^2 --> (m/s)^2
        vSquared_list = [const * dot(v, v) for v in self.iv]

        # get total kinetic energy
        E = sum(m * vSquared/2 for m, vSquared in zip(m_list, vSquared_list))

        # get tempurature from equal partition function
        numAtoms = self.getNumAtoms()
        return 2 * E / (3 * kb * numAtoms)

    def insert(self,
            other,
            selfOrigin = [0, 0, 0],
            otherOrigin = [0, 0, 0],
            angle = 0,
            axis = [0, 0, 1],
            ):
        """
        inserts another POSCAR structure
            * different from addMolecule in that overlapping atoms are deleted
            * intended to help insert small defects into large supercells
        other: POSCAR instance to be inserted (POSCAR)
        selfOrigin: location at which other will be in inserted (direct coords
            with respect to self's cell) (list of 3 floats)
        otherOrigin: location in other's cell that will align with selfOrigin
            (direct coords with respect to other's cell) (list of 3 floats)
            * generally keep at [0, 0, 0]
        theta: angle by which other is rotated (float)
        axis: axis about which other is rotated (list of three floats)
        """
        # UNDER CONTRUCTION: very odd rotations in inserted structure
        if stack()[1][3] == '<module>':
            self.updateTables()

        # place holder for parallel plane parameters
        coeff_tab = []

        # some parameters pertaining to self and other
        if type(selfOrigin) == int:
            selfOrigin = self.coords[selfOrigin - 1]
        if self.rep[0] == 'd' or self.rep[0] == 'D':
            selfCartOrigin = array(self.getCartOf(selfOrigin))
        else:
            selfCartOrigin = array(selfOrigin)

        other = other.copy() # don't manipulate other POSCAR

        if type(otherOrigin) == int:
            otherOrigin = other.coords[otherOrigin - 1]
        if other.rep[0] == 'd' or other.rep[0] == 'D':
            otherCartOrigin = array(other.getCartOf(otherOrigin))
        else:
            otherCartOrigin = array(otherOrigin)

        # shift other origin according to rotation
        shift = array([otherCartOrigin]) \
              - array(genRotate([otherCartOrigin.tolist()], angle, axis,
                      [0, 0, 0], [1]))

        # origin of other cell with respect to self's origin
        selfCartOrigin += shift[0] - otherCartOrigin
        selfOrigin = self.getDirOf(selfCartOrigin)
        
        # rotate other cell
        otherCell = deepcopy(other.cell)
        otherCell = array(genRotate(otherCell, angle, axis,
                          [0, 0, 0], [1,2,3]))

        # iterate through each pair of parallel planes
        for i in range(3):

            # cycle through permutations if i, j, and k
            j = (i + 1) % 3
            k = (i + 2) % 3

            # get triplet of points for each parallel plane
            v1, v2, v3 = otherCell[i], otherCell[j], otherCell[k]

            # get coefficients for plane equation
            #     a(x - x0) + b(y - y0) + c(z - z0) = 0
            coeff_list = cross(v1, v2)

            # get perpendicular direction and interplane distance
            perp = array(coeff_list) / norm(coeff_list)
            interplaneDist = abs(dot(perp, v3))
            print('interplane distance %s = %s, perp = %s'
                  %(i, interplaneDist, perp))
            coeff_tab.append((perp, v3, interplaneDist))

        # determine which coords are inside other's cell  
        rm_list = []
        for atom, coord in enumerate(self.coords):
            coord = self.getCartOf(coord)
            rmAtom = True

            # iterate through parallel plane pairs
            for perp, v3, interplaneDist in coeff_tab:
                dif1 = array(coord) - array(selfCartOrigin)
                dif2 = array(coord) - array(v3) - array(selfCartOrigin)
                dist1 = abs(dot(perp, dif1))
                dist2 = abs(dot(perp, dif2))

                # check if perp dist to either plane is greater than
                #     interplane dist
                if dist1 > interplaneDist or dist2 > interplaneDist:
                    rmAtom = False
                    break
            if rmAtom:
                rm_list.append(atom + 1)

        # remove overlapping atoms
        print('replacing atoms: %s' %rm_list)
        self.remove(rm_list)
        
        # add other POSCAR
        other.cell = otherCell
        self.addMolecule(other, selfOrigin)

    def invert(self, origin = [.5, .5, .5], atom_list = 'all'):
        """
        Inverts specified atoms about a speficied origin
        origin: a site (VESTA) or list of direct coordinates
            (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(origin) == int:
            origin = self.coords[origin - 1][:]

        if atom_list == 'all':
            atom_list = [n + 1 for n in range(self.getNumAtoms())]

        self.coords = genInvert(self.coords, origin, atom_list)

    def makeBigPrimCell(self, num_pts, layer = True, catchAngle = False):
        """
        resizes and rotates primitive cell to fit specified number of
            lattice points
            * requires square or hexagonal symmetry
        num_pts: number lattice points to be contained in cell (pos int)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # determine angle of symmetry
        dot_prod = abs(dot(self.cell[0], self.cell[1])/norm(self.cell[0])\
                           /norm(self.cell[1]))
        theta = arccos(-dot_prod) # ensure theta > pi / 2
        print("lattice angle = %s" %degrees(theta))

        # find closest number of lattice points that can fit in the cell
        super_len = int(ceil(sqrt(num_pts)))
        pair_list = [(n, m) for n in range(super_len + 1)\
                     for m in range(super_len + 1)]
        pair_list = list(set(pair_list))
        dim_list = [pair for pair in pair_list if pair[0] >= pair[1]]   
        num_list = [int(round(n**2 + m**2 - 2*n*m*cos(theta)))\
                    for n, m in dim_list]

        if num_pts not in num_list:
            index = (abs(array(num_list) - num_pts)).argmin()
            num_pts = num_list[index]
            print('fitting %s points into cell' %num_pts)
        else:
            index = num_list.index(num_pts)

        n, m = dim_list[index]
        print("new lattice vector in terms of old basis (%s, %s)" %(n, m))

        # determine angle of rotation
        if 0 not in (n, m):
            alpha = degrees(arccos((n**2 + num_pts - m**2)\
                                   /(2*n*sqrt(num_pts))))
        else:
            alpha = 0
        print("rotating %s degrees" %alpha)

        # new cell size
        new_cell = array([vector[:] for vector in self.cell])
        new_cell[0] *= sqrt(num_pts)
        new_cell[1] *= sqrt(num_pts)

        # make supercell
        self.makeSuperCell([2*super_len, 3*super_len, 1])
        
        # rotate coords
        self.rotate(alpha, axis = [0, 0, 1], origin = [0, 1/3, 0])

        # change cell size
        self.changeCell(new_cell, origin = [1/num_pts**2, 1/3, 0],
                        strip = True)
        self.strip()
        self.removeDoubles()

        # scale KPOINTS
        if self.ktable != None:
            for n in range(2):
                self.kmesh[n] = int(round(self.kmesh[n] / sqrt(num_pts)))

        if catchAngle:
            return alpha

    def makeCartesian(self):
        """
        Transforms self.coords to cartesian representation
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if self.rep[0] == 'c' or self.rep[0] == 'C':
            print('Coordinates are already cartesian.')
        else:
            if stack()[1][3] == '<module>':
                self.updateTables()

            for n in range(self.getNumAtoms()):
                x = array(self.coords[n][:])
                self.coords[n] = dot(x, self.cell).tolist()
            self.rep = 'Cartesian'

    def makeDirect(self):
        """
        Tranforms coordinates, self.coords, to direct representation
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if self.rep[0] == 'd' or self.rep[0] == 'D':
            print('Coordinates are already direct.')
        else:
            if stack()[1][3] == '<module>':
                self.updateTables()

            for n in range(self.getNumAtoms()):
                x = array(self.coords[n][:])
                self.coords[n] = dot(x, self.cell_inv).tolist()
            self.rep = 'Direct'

    def makeInSym(self, origin, atom_list = 'all'):
        """
        Adds atoms to produce inversion symmetry
        origin: a site (VESTA) or list of direct coordinates
            (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        copy = self.copy()
        copy.invert(origin, atom_list)
        self.addMolecule(copy)

        self.removeDoubles()

    def makeRefSym(self, plane, origin, atom_list = 'all', minDist = 0.01):
        """Adds atoms to produce reflection symmetry.

        plane: miller indices for reflection plane (list of 3 ints)
        origin: a site or list of direct coordinates that lies on
            the reflection plane (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be reflected (list of ints)
        """
        # Under construction: not perfect because of removeDoubles
        if stack()[1][3] == '<module>':
            self.updateTables()

        copy = self.copy()
        copy.reflect(plane, origin, atom_list)
        self.addMolecule(copy)

        self.removeDoubles(minDist = minDist)

    def makeRotInSym(self, degree, axis, origin, atom_list = 'all'):
        """
        Adds atoms to produce rotational symmetry
        degree: symmetry is degree-fold (int > 1)
        axis: axis of rotation (list of 3 floats)
        origin: a site (VESTA) or list of direct coordinates that lies on
            the axis of rotation (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        angle = 360 / degree
       
        copy = self.copy() 
        
        for i in range(2*degree - 1):
            copy.invert(origin, atom_list)
            copy.rotate(angle, axis, origin, atom_list)

            self.addMolecule(copy)

        print(self.coords)
        self.removeDoubles()

    def makeRotSym(self, degree, axis, origin, atom_list = 'all'):
        """
        adds atoms to produce rotational symmetry
        degree: symmetry is degree-fold (int > 1)
        axis: axis of rotation (list of 3 floats)
        origin: a site (VESTA) or list of direct coordinates that lies on
            the axis of rotation (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        angle = 360 / degree
       
        copy = self.copy() 
        
        for n in range(degree - 1):
            rot = copy.copy()
            rot.rotate(angle*(n + 1), axis, origin, atom_list)
            self.addMolecule(rot)

        self.removeDoubles()

    def makeScrewSym(self, degree, axis, origin, atom_list = 'all'):
        """
        adds atoms to produce rotational symmetry
        degree: symmetry is degree-fold (int > 1)
        axis: axis of rotation (list of 3 floats)
        distance: fraction of unit cell translated per rotation (float)
        origin: a site (VESTA) or list of direct coordinates that lies on
            the axis of rotation (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        angle = 360 / degree

        translation = array(axis) / norm(axis) / degree

        copy = self.copy() 
        
        for i in range(2*degree - 1):
            copy.translate(translation)
            copy.rotate(angle, axis, origin, atom_list)

            self.addMolecule(copy)

        self.bringIntoCell()

    def makeSuperCell(self, dim, glide = [0.0, 0.0, 0.0], screw = 0):
        """
        creates supercell lattice of specified dimenstion
        dim: dimensions of supercell (list of 3 floats)
        glide: glide vector between repetitions in z-direction
            (list of 3 floats)
        screw: screw angle between repetitions in z-direction (float)
        """
        # UNDER CONSTRUCTION: rotate velocities with screw axis
        if stack()[1][3] == '<module>':
            self.updateTables()

        # scale cell size up and coordinates down
        #     (shorter than using self.resizeVac)
        for vector in range(3):
            self.cell[vector] *= dim[vector]

            for atom in range(len(self.coords)):
                self.coords[atom][vector] *= 1 / dim[vector]

        self.cell_inv = inv(self.cell)

        # original initial velocities
        #     (already in cartesian, don't have to scale)
        iv_list = [iv[:] for iv in self.iv]
        sd_list = [sd[:] for sd in self.sd]

        # glide
        glide, dim = array(glide), array(dim)
        glide = glide / dim # /= doesn't work on integers for some reason

        # extrude unit cell coordinates in all directions to fit expanded cell
        range_dict = self.getSpecRanges()
        coords = [coord[:] for coord in self.coords]
        for spec in range_dict:
            first, last = range_dict[spec]
            spec_coords = array([coord[:] for coord in coords[first: last]])
            spec_ivs = [iv[:] for iv in iv_list[first: last]]
            spec_sds = [sd[:] for sd in sd_list[first: last]]
            index = self.specs.index(spec)
            for a in range(dim[0]):
                for b in range(dim[1]):
                    for c in range(dim[2]):
                        if a != 0 or b != 0 or c != 0:

                           # glide translation
                            g = glide * c
                            new_coords = [list(coord + [a/dim[0] + g[0],\
                                                        b/dim[1] + g[1],\
                                                        c/dim[2] + g[2]])\
                                          for coord in spec_coords]

                            # screw rotation
                            atom_list = [n + 1 for n in range(len(new_coords))]
                            new_coords = genRotate(new_coords, screw * c,
                                [0, 0, 1], [.5, .5, .5], atom_list)

                            self.add(spec, new_coords, iv_list=spec_ivs,
                                     sd_list = spec_sds)

        # scale KPOINTS
        if self.ktable != None:
            for n in range(3):
                self.kmesh[n] = int(round(self.kmesh[n] / dim[n]))

    def matchOrderTo(self, other, threshold = 1):
        """
        matches order of coordinates to those of another POSCAR instance
        threshold: max dist in Å for which an atomis recognized as
            the same atom (float)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # make copy of coordinate lists (mostly for legibility)
        selfCoords = deepcopy(self.coords)
        otherCoords = deepcopy(other.coords)

        # empty POSCAR in which final positions will be added in the
        #     correct order
        new = self.copy()

        # list of coord indices
        selfAtoms = [n + 1 for n in range(self.getNumAtoms())]
        new.remove(selfAtoms)

        # identify migrating atom as that which cannot be matched
        mismatch = False

        # find nearest atoms between self and other POSCARs
        for otherAtom, otherCoord in enumerate(other.coords):
            otherCoord = array(otherCoord)
            minDist = norm(other.cell) # upper bound for interatomic distance
            minCoord = None
            minAtom = None

            # find atom closest to self coord
            for selfAtom, selfCoord in zip(selfAtoms, selfCoords):
                selfCoord = array(selfCoord)
                difCoord = otherCoord - selfCoord

                # Only consider nearest image
                for n in range(3):
                    difCoord[n] -= round(difCoord[n])
                    
                # dif becomes new minDist if it is smaller
                dif = norm(self.getCartOf(difCoord))
                if dif < minDist:
                    minDist = dif
                    minCoord = selfCoord
                    minAtom = selfAtom

            # identify the atom that migrated
            if minDist > threshold:
                mismatch = True
                migAtom = otherAtom + 1
                print('migrating atom: %s' %migAtom)

            # add coordinates to new POSCAR in the same order as thay appear
            # in other, and remove coordinates from selfCoords list
            else:
                selfAtoms.remove(minAtom)
                newCoord = minCoord.tolist()
                selfCoords.remove(newCoord)
                newSpec = self.getSpecOf(minAtom) # species from self
                new.add(newSpec, newCoord)

        # insert migrating atom in correct position
        if mismatch:
            migSpec = other.getSpecOf(migAtom)
            new.add(migSpec, selfCoords[0])
            lastAtom = new.getSpecRanges()[migSpec][1]
            new.coords.insert(migAtom - 1, new.coords.pop(lastAtom - 1))

        # update self coords, specs, and pops
        self.coords = deepcopy(new.coords)
        self.specs = deepcopy(new.specs)
        self.pops = deepcopy(new.pops)
      

    def prepAdsorptionLandscape(self, species, corner1, corner2, ncolumns = 7, minDist = 2.25):
        """
        prepares POSCAR files for which adsorbed atom 
        species: Atomic symbol of atom being adsorbed (str)
        corner{1,2}: (direct) coordinates specifying corners of landscape
            (list of three of floats)
        ncolumns: number of intermediate positions between corners
        minDist: minumum interatomic distance (Å) between adsorbed atom and
            surface (float)
        """
        # UNDER CONSTRUCTION: ONLY PREPARES FOR 1D "LANDSCAPE"
        # make sure first two lattice vectors are on x, y plane
        self.alignCell()

        #linearly interpolate x, y coords between corner1 and corner2
        corner1 = array(corner1)
        corner2 = array(corner2)
        diff_ar = corner2 - corner1

        # output POSCAR directories and movie file
        dir_list = ['%02d' %n for n in range(ncolumns + 2)]
        movie_t3 = []

        print(corner1)
        for n in range(ncolumns + 2):

            # x, y coords shifted from corner1 by appropriate amount
            shift_ar = diff_ar * n / (ncolumns + 1)
            shift_ar += corner1

            # find atom closest to adatom
            adCoord = self.getCartOf(shift_ar)
            shortestDist = norm(self.cell)

            # hypothetically raise adatom above surface
            adCoord[2] += 2*minDist

            for m in range(self.getNumAtoms()):
                coord = array(self.getCartOf(self.coords[m]))
                dist = norm(coord - adCoord)
                if dist < shortestDist:
                    shortestDist = dist
                    closestCoord = coord
            
            # find smallest possible z coordinate of adsorbed atom
            r = norm(closestCoord[:2] - adCoord[:2])
            z = sqrt(minDist**2 - r**2)
            print('adCoord = %s,\nclosestCoord = %s' %(adCoord, closestCoord))
            print('shortestDist = %s, r = %s, z = %s' %(shortestDist, r, z))

            # add adsorbed atom to new POSCAR
            image = self.copy()
            image.sdOn()
            adCoord[2] = closestCoord[2] + z
            adCoord = self.getDirOf(adCoord)
            image.add(species, adCoord, sd_list = [['F', 'F', 'T']])
            image.write('%s/POSCAR' %dir_list[n])
            movie_t3.append(image.coords)
        print(corner2)
            
        # write movie in to xyz file
        with open('movie_scr.xyz', 'w') as f:
            range_dict = image.getSpecRanges()
            num_atoms = image.getNumAtoms()
            for n in range(len(movie_t3)):
                f.write('%s\nframe %02d' %(num_atoms, n))
                for spec in range_dict:
                    init, fin = range_dict[spec]
                    for coord in movie_t3[n][init: fin]:
                        f.write('\n %s' %spec)
                        for val in image.getCartOf(coord):
                            f.write('\t%s' %val)
                f.write('\n')

    def prepCellConvergence(self,
            scale_list = 'auto',
            layer = False,
            destination = '.',
            ):
        """
        prepares POSCAR and KPOINTS files to test convergence of KPOINTS
        cell_list: list of scales by which to multiply cell (list of floats)
            * if 'auto', set to [.96, .98, 1.00, 1.02, 1.04]
        destination: directory to which files are saved (str)
        """
        constant = norm(self.cell[0])
        volume = self.getCellVolume() # for naming directories

        if scale_list == 'auto':
            scale_list = [.96, .98, 1.00, 1.02, 1.04] 

        for scale in scale_list:
            POS = self.copy()
            if layer:
                POS.cell[0] *= scale
                POS.cell[1] *= scale
                directory = '%.5g' %(scale**2 * volume)
            else:
                POS.cell *= scale
                directory = '%.5g' %(scale**3 * volume)
#            directory = '%.5g' %(scale * constant)
#            POS.write(destination + '/' + directory + '/POSCAR_scr',
#                        destination + '/' + directory + '/KPOINTS')
            POS.write('%s/%s/POSCAR_scr' %(destination, directory),
                        '%s/%s/KPOINTS' %(destination, directory))
            POS.write('%s/%s/POSCAR' %(destination, directory),
                      writeKPOINTS = False)

    def prepKConvergence(self,
            k_list = 'auto',
            layer = False,
            kmin = 3,
            kmax = 12,
            destination = '.',
            ):
        """
        prepares POSCAR and KPOINTS files to test convergence of KPOINTS
        k_list: list of kmeshes (list of lists of 3 ints)
            * if auto, uses self.getK_list to generate k_list
        layer: if True, and if k_list = 'auto', third mesh dimenstion = 1
            (bool)
        kmin: min number of kpoints along shortest axis (pos int)
        kmax: max number of kpoints along shortest axis (pos int)
        destination: directory to which files are saved (str)
        """
        if self.ktable == None:
            self.addKPOINTS()

        if k_list == 'auto':
            k_list = self.getK_list(kmin, kmax, layer)

        for mesh in k_list:
            POS = self.copy()
            POS.kmesh = mesh
            meshStr_list = ['{0:0>2}'.format(mesh[n]) for n in range(3)]
            directory = '{}x{}x{}'.format(*meshStr_list)
            POS.write('%s/%s/POSCAR_scr' %(destination, directory),
                        '%s/%s/KPOINTS' %(destination, directory))
            POS.write('%s/%s/POSCAR' %(destination, directory),
                    writeKPOINTS=False)

    def prepLAMMPS(self,
            # charge_list = [1, -0.5], # W and S in WS2
            charge_list = [0, 0],
            outfile = 'auto',
            ):
        """
        writes LAMMPS structure file based on atom coordinates
        charge_list: list of charges on each atom (list of floats)
            * if list-length = number of atoms, charge is specified
                for each atom
            * if list-length = number of species, charge is specified
                for each species
            * if None, all atoms are neutral
        outfile: name of file to be written (str)
            * if 'auto': outfile = self.getName(LaTeX = False) + '.dat'
        """
        # automatically label outfile by chemical species
        if outfile == 'auto':
            outfile = '%s.dat' %self.getName(LaTeX=False)

        if '/' in outfile:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

        # align cell so that a1 has no y, z component and a2 has no z component
        self.alignCell()

        # properties of self
        numAtoms = self.getNumAtoms()
        specs = deepcopy(self.specs)
        cell = deepcopy(self.cell)

        with open(outfile, "w") as f:
            f.write('# %s\n\n' %self.getName())
            f.write('%s atoms\n' %numAtoms)
            f.write('%s atom types\n\n' %len(specs))

            # box
            a, b, c = [norm(vec) for vec in cell]
            xy = cell[1][0]
            xz = cell[2][0]
            yz = cell[2][1]

            f.write('0 %s xlo xhi\n' %a)
            f.write('0 %s ylo yhi\n' %b)
            f.write('0 %s zlo zhi\n' %c)
            f.write('%s %s %s xy xz yz\n\n' %(xy, xz, yz))

            # species' masses
            f.write('Masses # g/mol\n\n')

            spec_dict = {}
            for n, spec in enumerate(specs):
                mass = p_dict[spec][1]
                f.write('%s %s # %s\n' %(n + 1, mass, spec))
                spec_dict[spec] = n + 1

            # coordinates and charges
            f.write('\nAtoms # no. species, charge, x, y, z\n\n')

            # assign species to each coordinate
            spec_list = []
            n = 1
            for pop in self.pops:
                for i in range(pop):
                    spec_list.append(n)
                n += 1 

            # assign charge to each coordinate
            if charge_list == None:
                charge_list = [0.0 for n in range(numAtoms)]

            elif len(charge_list) == len(specs):
                newCharge_list = []
                for pop, charge in zip(self.pops, charge_list):
                    for i in range(pop):
                        newCharge_list.append(charge)
                charge_list = deepcopy(newCharge_list)

            # write atom no., spec, charge, x, y, z
            for n, spec, charge, coord in zip(range(numAtoms), spec_list,
                                              charge_list, self.coords):
                if self.rep[0] == 'd' or self.rep[0] == 'D': 
                    coord = self.getCartOf(coord)
                x, y, z = coord
                f.write('%s %s %s %s %s %s\n' %(n + 1, spec, charge, x, y, z))

            # write velocities
            f.write('\nVelocities # Å/fs\n\n')
            for n, iv in enumerate(self.iv):
                iv = array(iv) # Å/fs
                vx, vy, vz = iv 
                f.write('%s %s %s %s\n' %(n + 1, vx, vy, vz))

    def prepLayerConvergence(self,
            l_list,
            glide = [0, 0, 0],
            screw = 0,
            depth = None,
            vac = 0,
            destination = '.',
            direction = 3,
            addBottomSurface = False):
        """
        prepares POSCAR and KPOINTS files to test convergence of the number of
            layers in a surface calculation
        l_list: list number of layers (list of lists of 3 ints)
        glide: glide vector between repetitions in z-direction (float)
        depth: depth to which atoms will be allowed to relax (pos int or None)
        vac: vacuum height above top layer in Angstroms (postive float)
        destination: directory to which files are saved (str)
        direction: lattice direction in which to add layers (1, 2, or 3)
        """
        # NEEDS TESTING: automatically produce two identical surfaces
        for num_layers in l_list:
            POS = self.copy()

            # make supercell
            super_cell = [1, 1, 1]
            super_cell[direction - 1] = num_layers

            # add layers in z-direction
            POS.makeSuperCell(super_cell, glide, screw)

            # add bottoms surface
            if addBottomSurface:
                layer_dict = POS.getLayers()
                topSites = [tup[1] for tup in layer_dict[len(layer_dict)]]
                BotPOS = POS.copy()
                BotPOS.translate([0, 0, -1], topSites)
                POS.addMolecule(BotPOS)
                POS.resizeVac([None, None, (num_layers + 1) / num_layers],
                    operation = 'm', origin = [1,1,1])
                POS.removeDoubles()

            # add vacuum
            vac_size = [None, None, None]
            vac_size[direction - 1] = vac
            POS.resizeVac(vac_size, operation = 'add')

            # enable only top and bottom layers
            if depth != None:
                POS.sdOn()
                POS.sdMakeFalse()

            # enable sd on specified number of top and bottom layers
            if type(depth) == int:
                POS.sdOn()
                POS.sdMakeFalse()
                layer_dict = POS.getLayers()
                layer_list = [n + 1 for n in range(len(layer_dict))]

                true_layers = [layer_list[n] for n in range(depth)] + \
                              [layer_list[n] for n in range(-1, -depth -1, -1)]
                true_atoms = []
                for layer in true_layers:
                    true_atoms += layer_dict[layer]
                true_list = [tup[1] for tup in true_atoms]
                POS.sdMakeTrue(true_list)

            # enable sd on layers within a speficied depth
            elif type(depth) == float:
                height_list = [coord[direction - 1] for coord in POS.coords]
                cart_height_list = []
                for height in height_list:
                    height_vector = [0, 0, 0]
                    height_vector[direction - 1] = height
                    cart_height_list.append(
                            POS.getCartOf(height_vector)[direction - 1])

                top = max(cart_height_list)
                bottom = min(cart_height_list)
                true_list = []
                for n in range(len(cart_height_list)):
                    height = cart_height_list[n]
               
                    if top - height <= depth or height - bottom <= depth:
                        true_list.append(n + 1)
                print(true_list)
                POS.sdMakeTrue(true_list)

            # write to directories labelled by number of layers
            directory = '{0:0>2}'.format(num_layers)
            POS.write(destination + '/' + directory + '/POSCAR_scr',
                        destination + '/' + directory + '/KPOINTS')

    def prepNEB(self,
            other,
            number,
            fixedLengths = [],
            alongBondCenter = False,
            fixedAtoms = [],
            align = True,
            alignment = 'all',
            matchOrder = False,
            threshold = 1,
            ):
        """
        prepares image POSCARs for NEB calculation. Also writes movie.xyz file
        other: instance of POSCAR class for final state (POSCAR)
        number: number of images (pos int)
        fixedLengths: indicates which bond lengths are kept constant
            (list of tuple pairs)
            * atom1 position is linearly interpolated, atom2 moves
                correspondingly
        alongBondCenter: if True, bond center of fixed bond move linearly
            (bool)
        fixedAtoms: atoms (as in VESTA) that need to be fixed (list of pos int)
        align: if True, aligns initial and final states before interpolating
            (bool)
        alignement: 'all' or 'cm' (str)
        matchOrder: if True, matches order of other coordinates to self
        threshold: max dist in Å for which an atom is recognized as the same
            atom (float)
        """
        # UNDER CONSTRUCTION: allow for rotation of bonds about an atom
        # match order of other coordinates to self
        if matchOrder:

            # don't manipulate original copy of other POSCAR
            other = other.copy()
            other.matchOrderTo(self, threshold = threshold)

        # align self and other
        if align:

            # align center of masses of initial and final POSCARs
            if alignment[0] == 'c' or alignment[0] == 'C':
                cm1 = self.getCM()
                cm2 = other.getCM()
                dif_ar = cm1 - cm2

            # shifts final POSCAR by average displacement
            elif alignment[0] == 'a' or alignement[0] == 'A':
                coords1 = deepcopy(self.coords)
                coords2 = deepcopy(other.coords)
                dif_ar = zeros(3) # to shift initial and final positions
                for coord1, coord2 in zip(coords1, coords2):
                    for comp in range(3):
                        dif = coord1[comp] - coord2[comp]

                        # distance between nearest periodic image
                        dif -= round(dif)
                        dif_ar[comp] += dif
                dif_ar /= self.getNumAtoms()

            # don't manipulate original copy of other POSCAR
            other = other.copy()
            print('shifting %s by %s' %(other, dif_ar))
            other.translate(dif_ar)

        # convert fixed lengths to a table
        if len(fixedLengths) > 0:
            if type(fixedLengths[0]) == int:
                fixedLengths = [fixedLengths]

        # write initial and final states
        self.write('00/POSCAR')
        other.write('%02d/POSCAR' %(number + 1)) 

        # linear interpolation betwen init and final states
        coords1_ar = array([coord[:] for coord in self.coords])
        coords2_ar = array([coord[:] for coord in other.coords])
        dif_tab = []
        for coord1, coord2 in zip(coords1_ar, coords2_ar):
            dif_list = []
            for comp in range(3):
                dif = coord2[comp] - coord1[comp]
                dif -= round(dif)
                dif_list.append(dif)
            dif_tab.append(dif_list)
        dif_ar = array(dif_tab)
#        dif_ar = coords2_ar - coords1_ar

        # list of directories into which images are written
        dir_list = ['%02d' %(n + 1) for n in range(number)]

        movie_t3 = [coords1_ar]
        for n in range(number):

            # shift image coords by appropriate amount
            shift_ar = dif_ar * (n + 1) / (number + 1)
            shift_ar += coords1_ar

            # keep certain atoms fixed
            for atom in fixedAtoms:
                shift_ar[atom - 1] = self.coords[atom - 1][:]

            # create image as instance of POSCAR class
            image = self.copy()
            image.coords = [list(coord) for coord in shift_ar]

            # keep certain bondlengths constant; rotate about center of bond
            for atom1, atom2 in fixedLengths:
                imageAxis = self.getCartOf(array(shift_ar[atom2 - 1])\
                        - array(shift_ar[atom1 - 1]))

                startAxis = self.getCartOf(self.getAxis(atom1, atom2))
                imageDist = norm(imageAxis)
                startDist = norm(startAxis)
                difDist = startDist - imageDist
                unit = imageAxis / imageDist
                if alongBondCenter:
                    tran = unit * difDist / 2
                    image.translate(tran, atom_list = atom2, cartesian = True)
                    image.translate(-tran, atom_list = atom1, cartesian = True)
                else:
                    tran = unit * difDist
                    image.translate(tran, atom_list = atom2, cartesian = True)
#                print(image.getDist(atom1, atom2))

            # write to image directories
            image.bringIntoCell()
            image.write('%s/POSCAR' %dir_list[n])
            movie_t3.append(image.coords)
        movie_t3.append(coords2_ar)

        # write movie in to xyz file
        with open('movie_scr.xyz', 'w') as f:
            range_dict = self.getSpecRanges()
            num_atoms = self.getNumAtoms()
            for n in range(len(movie_t3)):
                f.write('%s\nframe %02d' %(num_atoms, n))
                for spec in range_dict:
                    init, fin = range_dict[spec]
                    for coord in movie_t3[n][init: fin]:
                        f.write('\n %s' %spec)
                        for val in self.getCartOf(coord):                 
                            f.write('\t%s' %val)
                f.write('\n')

    def prepSputter(self,
            atom,
            En,
            outfile = 'POSCAR_sput',
            ):
        """
        prepares POSCAR file with IVs initialized for sputter MD
        atom: atom being sputtered starting from 1 (pos int)
        En: recoil energy in eV (pos float)
        outfile: POSCAR file to be written (str)
        """
        POS = self.copy()

        species = POS.getSpecOf(atom)
        M = p_dict[species][1] * mp
        vn = (2 * En / M)**(1/2) * c / 100000   # Å/fs
        POS.setIV((0, 0, vn), atom)

        POS.write(outfile)

    def prepSputterSeries(self,
            atom,
            EVac,  # eV
            interval = .1,  # eV
            outfile = 'POSCAR_sput',
            ):
        """
        prepares POSCAR file with IVs initialized for sputter MD
            atom: atom being sputtered starting from 1 (pos int)
        En: vacancy formation energy in eV (pos float)
        interval: interval between recoil energy tests (pos float)
        outfile: POSCAR file to be written (str)
        """
        Emin = round(EVac, 1) - 0.3  # eV
        En_list = arange(Emin, Emin + 7*interval, interval)

        for En in En_list:
            POS = self.copy()
            species = POS.getSpecOf(atom)
            M = p_dict[species][1] * mp
            vn = (2 * En / M)**(1/2) * c / 100000   # Å/fs
            POS.setIV((0, 0, vn), atom)
            POS.write('%.2f/%s' %(En, outfile))
            POS.write('%.2f/POSCAR' %En)

    def prepVacConvergence(self,
            width_list = 'auto',
            origin = [.5, .5, .5],
            destination = '.',
            asym = True,
            ):
        """
        prepares POSCAR files to test convergence of vacuum for molecules
        width_list: list of vacuum distances to test (list of floats)
        origin: origin about which cell is rescaled (list of 3 floats)
        destination: directory to which files are saved (str)
        asym: if True, cell dimensions are asymmetrical
        """
        if width_list == 'auto':
            width_list = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

        for n in width_list:
            POS = self.copy()

            n_str1 = '{0:0>2}'.format(n)
            if asym:

                # break symmetry for molecules
                POS.resizeVac([n, n + 1, n + 2], origin = origin)
                n_str2 = '{0:0>2}'.format(n + 1)
                n_str3 = '{0:0>2}'.format(n + 2)
                directory = '%sx%sx%s' %(n_str1, n_str2, n_str3)
            else:
                POS.resizeVac([n, n, n], origin = origin)
                directory = '%sx%sx%s' %(n_str1, n_str1, n_str1)

            POS.write(destination + '/' + directory + '/POSCAR_scr',
                        destination + '/' + directory + '/KPOINTS')
            POS.write(destination + '/' + directory + '/POSCAR')

    def prepZConvergence(self,
            z_list = 'auto',
            destination = '.',
            operation = 'set equal',
            depth = None,
            direction = 3,
            origin = [.5,.5,.5],
            ):
        """
        prepares POSCAR files to test convergence of vacuum distance between
            layers
        z_list: list of vacuum distances to test (list of floats)
        destination: directory to which files are saved (str)
        operation: operation by which z_list transforms cell height
            ('multiply or add')
        depth: depth to which atoms will be allowed to relax
            if type(depth) == int, sd is True for depth # of layers
            if type(depth) == float, sd is True within depth angstroms from
                surface or edge
            if depth == None, selective dynamics is not changed
        direction: lattice direction in which to add vacuum (1, 2, or 3)
        """
        # automatically generate cell heights
        if z_list == 'auto':
            z_list = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

        # Turn on sd for atoms belonging to top and bottom layers
        if type(depth) == int:
            layer_dict = self.getLayers()
            layer_list = [n + 1 for n in range(len(layer_dict))]

            true_layers = [layer_list[n] for n in range(depth)] + \
                          [layer_list[n] for n in range(-1, -depth - 1, -1)]
            true_atoms = []
            for layer in true_layers:
                true_atoms += layer_dict[layer]
            true_list = [tup[1] for tup in true_atoms]

#        if type(depth) == float:
#            pass

        # if third lattice vector is not parallel to z-direction
        ratio = norm(self.cell[direction - 1])\
                / self.cell[direction - 1][direction - 1]
        
        # write resized POSCARs
        for n in z_list:
            POS = self.copy()
            if type(depth) == int:
                POS.sdOn() # enable layers on top and bottom
                POS.sdMakeFalse()
                POS.sdMakeTrue(true_list)
            directory = '{0:0>2}'.format(n)
            vac_size = [None, None, None]
            vac_size[direction - 1] = n * ratio
            POS.resizeVac(vac_size, operation, origin)
            POS.write(destination + '/' + directory + '/POSCAR_scr',
                        destination + '/' + directory + '/KPOINTS')
            POS.write(destination + '/' + directory + '/POSCAR')

    def reflect(self, plane, origin = [.5, .5, .5], atom_list = 'all'):
        """
        reflects reflects specified atoms about specified plane
        plane: miller indices for reflection plane (list of 3 ints)
        origin: a site or list of direct coordinates that lies on the
            reflection plane (int of list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be reflected (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(origin) == int:
            origin = self.coords[origin - 1]

        rec_cell = self.getRecip()  # reciprocal lattice vectors
        # origin in cartesian coordinates
        origin = dot(origin, self.cell).tolist()

        if atom_list == 'all':
            atom_list = [n + 1 for n in range(self.getNumAtoms())]

        for atom in atom_list:
            self.coords[atom - 1] = self.getCartOf(self.coords[atom - 1])

        self.coords = genReflect(self.coords, rec_cell, plane, origin,
                atom_list)

        for atom in atom_list:
            self.coords[atom - 1] = self.getDirOf(self.coords[atom - 1])

    def relocate(self, atom_list, dest_list):
        """
        translates specified atoms to specified locations in direct coordinates
        atom_list: atoms, indexed as in VESTA, to be moved (list of ints)
        dest_list: list of atom desinations in fractional coordinates
            (list of lists)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()
        
    def remove(self, atom_list):
        """
        removes atoms in atom_list
        atom_list: atoms, indexed as in VESTA, to be removed (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(atom_list) == int:
            atom_list = [atom_list]
        else:
            atom_list = list(set(atom_list)) # remove duplicates

        # remove atoms from self.coords in descending order to maintain
        #     atom numbering
        atom_list.sort(reverse = True)

        for atom in atom_list:
            del(self.coords[atom - 1])
            del(self.sd[atom - 1])
            del(self.iv[atom - 1])

        # substract atoms from self.pops
        self.subtractFromPops(atom_list)

    def removeDoubles(self, minDist=.01):
        """Removes atoms that are unphysically close to another atom.

        minDist: interatomic distance in Å below which atoms are considered
            too close (float)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        range_dict = self.getSpecRanges()

        # cartesian coordinates
        coords = [self.getCartOf(coord) for coord in self.coords]

        # find pairs of atoms that are unphysicially close together
        rm_list = []
        start = 0  # track indices to avoid double counting pairs
        
        for i in range(len(coords)):
            for j in range(start, len(coords)):
                if i != j:
                    delta = [coords[i][n] - coords[j][n] for n in range(3)]
                    delta2 = [coords[i][n] - coords[j][n]\
                           - round(coords[i][n]) + round(coords[j][n])\
                             for n in range(3)]
                    dist = norm(delta)
                    dist2 = norm(delta2)
                    if dist < minDist or dist2 < minDist:
                        # only remove atoms of same species
                        spec_i, spec_j = self.getSpecOf(i + 1),\
                                self.getSpecOf(j + 1)
                        if spec_i == spec_j:
                            rm_list.append(i + 1)
#                        else:
#                            print(
#        'WARNING: %s atom at %s and %s atom at %s are unphysically close together'
#        %(spec_i, i + 1, spec_j, j + 1))

            start += 1 # avoid double counting

        self.remove(rm_list)

    def removeExterior(self):
        """ removes atoms that are outside of the cell """
        if stack()[1][3] == '<module>':
            self.updateTables()

        rm_list = []
        for i in range(len(self.coords)):
            for comp in self.coords[i]:
                if comp < 0 or comp > 1:
                    rm_list.append(i + 1)
                    break

        self.remove(rm_list)

    def removeSpecs(self, spec_list):
        """
        removes selected species from POSCAR
        spec_list: list of specs to be removed (list of str)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(spec_list) == str:
            spec_list = [spec_list]

        range_dict = self.getSpecRanges()

        rm_list = []
        for spec in range_dict:
            if spec in spec_list:
                first, last = range_dict[spec]
                rm_list += list(range(first + 1, last + 1))

        print(rm_list)
        self.remove(rm_list)
        
    def reorderSpecs(self, newSpec_list):
        """
        reorders specs
        newSpec_list: list of specs in desired order (list or str)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # old ranges of coordinate indices for each species
        specRange_dict = self.getSpecRanges()

        # create new list of coords species by species
        newCoords_list = []
        for newIndex, spec in enumerate(newSpec_list):

            # exit command if a species in unrecognized
            if spec not in self.specs:
                print('%s not in POSCAR' %spec)
                return

            # add new coordinates in order of newSpecies_list
            else:
                firstIndex, lastIndex = specRange_dict[spec]
                for n in range(firstIndex, lastIndex):
                    newCoords_list.append(self.coords[n][:])

        # replace old coordinates
        self.specs = deepcopy(newSpec_list)
        self.coords = deepcopy(newCoords_list)
 
    def reorderVecs(self, newOrder_list):
        """Reorders lattice vectors and coordinate components.

        newOrder_list: new permutation of lattice vectors (list of 0,1,2)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        # holders for new cell and coordinates
        newCell = []
        newCoords = [[] for coord in range(self.getNumAtoms())]

        # reorder cell and coordinats components
        for vec in newOrder_list:

            # reorder cell
            newCell.append([])
            for comp in newOrder_list:
                newCell[-1].append(self.cell[vec][comp])

            # reorder coordinate components
            for coord in range(self.getNumAtoms()):
                newCoords[coord].append(self.coords[coord][vec])

        # update instance
        self.cell = array(newCell)
        self.coords = newCoords
        
    def reshapeCell(self, new_cell):
        """
        reshapes cell without changing atom positions
        new_cell: (3x3 array of floats)
        """
        self.makeCartesian()
        self.cell = array(new_cell)
        self.makeDirect()
        
    def resizeVac(self,
            new_len_list,
            operation = 'set equal',
            origin = [0, 0, 0],
            ):
        """
        resizes the unit cell while inversely scaling the atom positions
        new_len_list: lengths of new lattice vectors in Angstroms
            (list of 3 floats)
            * if list element = None, does not resize the corresponding vector
        operation: operation by which z_list transforms cell height
            ('multiply or add')
        origin: center around which coordinates are scaled
	"""
        if stack()[1][3] == '<module>':
            self.updateTables()

        if len(self.coords) > 1:  # avoid empty cell error

            # center coords on origin
            coords = array(self.coords) - array(origin)

        else:
            coords = []

        for n in range(3):
            if new_len_list[n] != None:
                if operation[0] == 'm':
                    scale = new_len_list[n]
                elif operation[0] == 'a':
                    old_len = norm(self.cell[n])
                    scale = (old_len + new_len_list[n]) / old_len
                else:
                    old_len = norm(self.cell[n])
                    scale = new_len_list[n] / old_len

                for i in range(3):         
                    self.cell[n][i] *= scale  # multiply cell by scaling factor

                for j in range(len(coords)):
                    coords[j][n] *= 1/scale  # divide coords by scaling factor

        if len(coords) > 1:
            self.coords = (coords + array(origin)).tolist()

        self.cell_inv = inv(self.cell)  # update inverse cell as well

    def rotate(self,
            angle,
            axis = [0, 0, 1],
            cartesian = False,
            origin = [.5, .5, .5],
            atom_list = 'all'
            ):
        """
        rotates specified atoms about a speficied axis
        angle: angle in degrees (float)
        axis: axis of rotation (list of 3 floats)
            * Miller indices by default
        cartesian: if True, axis is assumed to be in cartesian coordinates
            (bool)
        origin: a site (VESTA) or list of direct coordinates that lies on the
            axis of rotation (int or list of 3 floats)
        atom_list: atoms, indexed as in VESTA, to be rotated (list of ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(origin) == int:
            origin = self.coords[origin - 1][:]

        # axis and origin in cartesian coordinates
        if not cartesian:
            axis = dot(axis, self.cell).tolist()
        origin = dot(origin, self.cell).tolist()

        if atom_list == 'all':
            atom_list = [n + 1 for n in range(self.getNumAtoms())]

        for atom in atom_list:
            self.coords[atom - 1] = self.getCartOf(self.coords[atom - 1])

        self.coords = genRotate(self.coords, angle, axis, origin, atom_list)

        for atom in atom_list:
            self.coords[atom - 1] = self.getDirOf(self.coords[atom - 1])

    def roundCoords(self, degree = 5, atom_list = 'all', components = 'all'):
        """
        rounds specified coordinates to the specified degree
        degree: degree of rounding
            if 'auto', rounds all coords of 5 or more repeating decimals
        atom_list: atoms (starting from 1) whose coordinates will be rounded
            (list of pos ints)
        components: components (starting from 1) that will be rounded
            (list of pos ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if atom_list == 'all':
            atom_list = [n for n in range(self.getNumAtoms())]
        else:
            atom_list = [n - 1 for n in atom_list]

        if components == 'all':
            components = [n for n in range(3)]
        else:
            components = [n - 1 for n in components]

        degree -= 1 # number of consecutive pairs = number of repetitions - 1

        for i in atom_list:
            for j in components:
                comp = self.coords[i][j]
                comp_str = str(comp) + ' ' # to iterate from 0 to n + 1
                if 'e' in comp_str:
                    self.coords[i][j] = 0.0
                else:
                    consecutive = n = 0
                    while consecutive < degree and n < len(comp_str) - 1:
                        if comp_str[n] == comp_str[n + 1]:
                            consecutive += 1
                        else:
                            consecutive = 0
                        n += 1               

                    if consecutive == degree:
                        n -= consecutive
                        rep = comp_str[n]

                        # round .999 to 1
                        if int(rep) == 9:
                            new_comp = round(comp,
                                    n - int(floor(log10(abs(comp)))))
                            print('round up: %s to %s' %(comp, new_comp))
                            if new_comp == 1.0:
                                new_comp = 0.0
                            self.coords[i][j] = new_comp     

                        # round up
                        elif 4 < int(rep) < 9:
                            new_comp_str = comp_str[:n] + rep * (17 - n)\
                                    + str(int(rep) + 1)
                            print('round up: %s to %s'
                                    %(comp_str, new_comp_str))
                            self.coords[i][j] = float(new_comp_str)

                        # round down
                        else:
                            new_comp_str = comp_str[:n] + comp_str[n]\
                                    * (18 - n)
                            print('round down: %s to %s'
                                    %(comp_str, new_comp_str))
                            self.coords[i][j] = float(new_comp_str)

                    else:
                        self.coords[i][j] = float(comp_str)
                    
    def sdMakeFalse(self, atom_list = 'all', directions = 'all'):
        """ 
        disables all dynamics for specified atoms
        atom_list: list of atoms, al a VESTA (iterable of ints or 'all')
        directions: directions to disable
            (iterable containing any combination of 1,2,3)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if atom_list == 'all':
            atom_list = [n for n in range(self.getNumAtoms())]
        else:
            atom_list = [val - 1 for val in atom_list]

        for atom in atom_list:
            self.sd[atom] = ['F', 'F', 'F']

    def sdMakeTrue(self, atom_list = 'all', directions = 'all'):
        """
        enables all dynamics for all atoms
        atom_list: list of atoms, al a VESTA (iterable of ints or 'all')
        directions: directions to enable
            (iterable containing any combination of 1,2,3)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if atom_list == 'all':
            atom_list = [n for n in range(self.getNumAtoms())]
        else:
            atom_list = [val - 1 for val in atom_list]

        for atom in atom_list:
            self.sd[atom] = ['T', 'T', 'T']

    def sdOff(self):
        """ turns off selective dynamics ([] for all coords) """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if not self.SD:
            print('Selective dynamics is already off.')
        else:
            self.sd = [[] for coord in self.coords]
            self.SD = False

    def sdOn(self):
        """ turns on selective dynamics (['T', 'T', 'T'] for all coords) """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if self.SD:
            print('Selective dynamics is already on.')
        else:
            self.sdMakeTrue()
            self.SD = True

    def selectDepth(self, depth = 5.0, direction = True):
        """
        returns list of atoms that are within a specified depth from a
            surface or edge
        depth: depth from surface or edge
            if type == int, depth is number of layers
            if type == float, depth is distance in angstroms
        direction: miller indices of direction with respect depth is measured
            (tup of 3 floats)
        """
        # Find "top" coordinate in chosen direction
        dir_unit = norm(direction)
        depth_list = [dot(coord, dir_unit) for coord in self.coords]
        top = max(depth_list)
        
        true_list = []

        if type(depth) == float:
            cart_unit = self.getCartOf(dir_unit)
            dir_depth = depth / norm(cart_unit)
            for point in depth_list:
                if point < top - dir_depth:
                    true_list.append(index(point))

        if type(depth) == int:
            pass

        self.sdMakeTrue(true_list)

    def selectRec(self, origin, vertex):
        """Returns list of atoms (from 1) within desired parallelepiped.

        origin: first vertex of the parallelepiped (list of 3 floats)
        vertex: opposite vertex of the parallelepiped (list of 3 floats)
            ** All components of vertex > components of origin
        """
        atom_list = []
        for atom in range(len(self.coords)):
            add = True
            for dim in range(3):
                comp = self.coords[atom][dim]
                if comp < origin[dim] or comp > vertex[dim]:
                    add = False
                    break
            if add:
                atom_list.append(atom + 1)

        return atom_list

    def selectSpec(self, spec_list):
        """
        returns list of atoms (indexed a la VESTA) of desired species
        """
        if type(spec_list) == str:
            spec_list = [spec_list]

        range_dict = self.getSpecRanges()

        atom_list = []
        for spec in spec_list:
            site_list = [n + 1 for n in range(range_dict[spec][0],
                range_dict[spec][1]) ]
            atom_list += site_list

        return atom_list

    def selectSphere(self, center, radius):
        """
        returns list of atoms (indexed a la VESTA) within desired sphere
        """
        pass

    def setIV(self, v_tab, atom_list):
        """
        Sets initial velocities to specified atoms
        v_tab: tuple of velocities in A/fs (tup of lists of 3 floats)
        atom_list: list of atoms as indexed in VESTA (list of ints)
            orderings of v_tab and atom_list should match
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if type(v_tab[0]) != list and type(v_tab[0]) != tuple:
#        if type(v_tab[0]) == float or type(v_tab[0]) == int\
#        or type(v_tab[0]) == numpy.float64:
            v_tab = [v_tab]

        v_tab = [[float(comp) for comp in v] for v in v_tab]

        if atom_list == 'all':
            atom_list = [n for n in range(self.getNumAtoms())]
        elif type(atom_list) == int:
            atom_list = [atom_list - 1]
        else:
            atom_list = [atom - 1 for atom in atom_list]

        for n in range(len(atom_list)):
            self.iv[atom_list[n]] = v_tab[n]

    def sortCoords(self, column = 3):
        """
        sorts coords in increasing order of selected direction
        column: columns to be sorted (1, 2, or 3)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        range_dict = self.getSpecRanges()
        coords = self.coords[:]
        sd = self.sd[:]

        for spec in range_dict:
            lower, upper = range_dict[spec]
            spec_list = [(self.coords[n][column - 1], n)\
                    for n in range(lower, upper)]
            spec_list.sort()
            order_list = [tup[1] for tup in spec_list]
            for n in range(lower, upper):
                self.coords[n] = coords[order_list[n - lower]] 
                self.sd[n] = sd[order_list[n - lower]]

    def strip(self):
        """ removes atoms that are outside of cell """
        if stack()[1][3] == '<module>':
            self.updateTables()

        rm_list = []
        for index in range(len(self.coords)):
            for val in self.coords[index]:
                if val > 1 or val < 0:
                    rm_list.append(index + 1)
                    break

        self.remove(rm_list)
                
    def subtractFromPops(self, atom_list):
        """
        Subracts from elements in self.pops
        atom_list: atoms, as indexed in VESTA, to be removed from pops
            (list of ints)
        """
        range_dict = self.getSpecRanges()

        # subtract atoms from self.pops
        for atom in atom_list:
            for spec in range_dict:
                lower, upper = range_dict[spec]
                if atom > lower and atom <= upper:
                    self.pops[ self.specs.index(spec) ] -= 1

        # remove spec from self.specs if its population is zero
        # count backwards to maintain order
        for n in range(len(self.pops) - 1, -1, -1):
            if self.pops[n] < 1:
                del(self.pops[n], self.specs[n])
        
    def swap(self, site1, site2):
        """
        Swaps the position of two or more atoms
            * to prep a migration, swap the coords, then remove atoms
        sites: pair of atomic sites to switch (pos ints)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()
        
        self.coords[site1 - 1], self.coords[site2 - 1]\
                = self.coords[site2 - 1], self.coords[site1 - 1]

    def translate(self, translation, atom_list = 'all', cartesian = False):
        """
        Translates specified atoms in direct coordinates
        atom_list: atoms, indexed as in VESTA, to be moved (list of ints)  
            * if atom_list = 'all', translates all atoms
            * if atom_list = 'species', translates all atoms of that species
        translation: translation vector in fractional coordinates
            (list of 3 floats)
        cartesian: if true, translation read in cartesian coordinates
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if cartesian:
            self.makeCartesian()

        if atom_list == 'all':
            atom_list = [n + 1 for n in range(self.getNumAtoms())]

        if atom_list in self.specs:
            pass

        if type(atom_list) == int:
            atom_list = [atom_list]

        self.coords = genTranslate(self.coords, translation, atom_list)

        if cartesian:
            self.makeDirect()

    def translateTo(self,
            atom,
            newCoord,
            atom_list = 'all',
            cartesian = False,
            ):
        """
        Translates atoms such that one atom moves to a specified location
        atom_list: atoms, indexed as in VESTA, to be moved (list of ints)  
            * if atom_list = 'all', translates all atoms
            * if atom_list = 'species', translates all atoms of that species
        translation: translation vector in fractional coordinates
            (list of 3 floats)
        cartesian: if true, translation read in cartesian coordinates
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        if cartesian:
            self.makeCartesian()

        if atom_list == 'all':
            atom_list = [n + 1 for n in range(self.getNumAtoms())]

        if atom_list in self.specs:
            pass

        if type(atom_list) == int:
            atom_list = [atom_list]

        translation = (array(newCoord) - array(self.coords[atom - 1])).tolist()

        self.coords = genTranslate(self.coords, translation, atom_list)

        if cartesian:
            self.makeDirect()

    def undo(self):
        """ Undoes last change """
        self.__init__(self.table, self.ktable)

    def updateTables(self):
        """ Constructs table with up-to-date attributes """
        coordsAndSd = [self.coords[n] + self.sd[n]\
                for n in range(self.getNumAtoms()) ]

        self.table = [self.name[:]] + [self.scale[:]] \
            + [vec[:] for vec in self.cell.tolist()] + [self.specs[:]] \
            + [self.pops[:]] + [[self.rep]] \
            + coordsAndSd + [[]] + self.iv

        if self.SD:
            self.table.insert(7, ['Selective Dynamics'])

        if self.ktable != None:
            self.ktable = [self.kcomment[:]] + [[self.knumber]] \
                + [[self.ktype]] + [self.kmesh[:]] \
                + [self.kshift[:]]

#----------------------------- POSCAR subclasses ------------------------------

class POSCAR2D(POSCAR):
    """
    subclass of POSCAR whose reduced dimansionality requires unique functions
    """
    def __init__(self, table, num_layers, KPOINTS = None):
        """
        num_layers = number of layers (int)
        """
        POSCAR.__init__(self, table, KPOINTS)
        self.num_layers = num_layers

    def addlayer():
        """
        adds layer to the system
        """
        pass

#------------------------------------------------------------------------------

class POSCAR2HTMD(POSCAR):
    """
    subclass of 2dPOSCAR
    TMD with hexagonal symmetry
    """
    def makeTrefoil(self, center):
        """
        inserts unrelaxed trefoil defect into tmd lattice
        center: TM atom on which defect is centered, as labelled in VESTA (pos int)
        """
        if stack()[1][3] == '<module>':
            self.updateTables()

        nn_dict = self.getNN(center)
        nn1_list = [site_tup[1] for site_tup in nn_dict[1]]
        nn3_list = [site_tup[1] for site_tup in nn_dict[3]]
        self.rotate(60, (0,0,1), center, nn1_list)
        self.remove(nn3_list)

    def addlayer():
        """
        adds layer to the system
        """
        pass

#    def test(self):
#        print('executing test')
#        print(stack()[1][3])
#        if stack()[1][3] == '<module>':
#            print(stack()[1][3])
#            print('called from module')
#        else:
#            print(stack()[1][3])
#            print('called from function')
#
#    def stacktest(self):
#        print('executing stacktest')
#        self.test()

