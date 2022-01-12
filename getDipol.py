# get dipole moment from CHGCAR
# Anthony Yoshimura
# 01/28/18

from numpy import array, reshape, zeros, cross, dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from periodic import table as spec_dict

def getDipol(CHGCAR = 'CHGCAR', OUTCAR = 'OUTCAR', cartesian = False):
    """
    returns x, y, and z components of dipole moment in direct coordinates
        units = electrons * Angstrom (if cartesian)
    CHGCAR: CHGCAR file containing charge density (str)
    OUTCAR: OUTCAR file containing ZVAL values (str)
    cartesian: if True, dipole moment given in cartesian coordinates (bool)
    """
    zval_list = []
    with open(OUTCAR) as f:
        for line in f:
            if 'ZVAL' in line and 'POMASS' in line:
                zval_list.append(float(line.split()[-4]))
    print('valence numbers: %s' %zval_list)

    with open(CHGCAR) as f:

        # get lattice vectors and cell volume
        for n in range(2): f.readline()
        a1 = [float(val) for val in f.readline().split()]
        a2 = [float(val) for val in f.readline().split()]
        a3 = [float(val) for val in f.readline().split()]
        len1, len2, len3 = norm(a1), norm(a2), norm(a3)
        crossProduct = cross(a1, a2)
        cellVolume = abs(dot(a3, crossProduct))

        # get atomic populations
        spec_list = [spec for spec in f.readline().split()] 
        pop_list = [int(val) for val in f.readline().split()]

        # sum over nuclei contribution to dipole moment (ignoring core charges)
        dipMom = zeros(3)
        f.readline()
        coords_list = []
        for pop, zval in zip(pop_list, zval_list):
            for n in range(pop):
                coord = array([float(val) for val in f.readline().split()])
                dipMom += coord * zval

        # get number of grid points
        f.readline()
        ng_str = f.readline().split()
        ngx, ngy, ngz = [int(val) for val in ng_str]
        ng = ngx * ngy * ngz

        # reshape charge density data into 3d array
        chg_str = f.read()
        chg_list = chg_str.split()[:ng]
        chg1d = array([float(val) for val in chg_list])
        chg3d = reshape(chg1d, (ngz, ngy, ngx))

        # integrate r * rho over cell to obtain electron contribution to dipole moment
        dv = 1/ng
        dx, dy, dz = 1/ngx, 1/ngy, 1/ngz
        for x in range(ngx):
            for y in range(ngy):
                for z in range(ngz):
                    r = array([x*dx,y*dy,z*dz])
                    integrand = r * chg3d[z][y][x]
                    dipMom -= integrand * dv

        # transform to cartesian coordinates (Angstrom)
        if cartesian:
            cell = array(a1, a2, a3)
            dipMom = dot(dipMom, cell) * cellVolume

        return dipMom 

#---------------------------- SCRATCH ------------------------------------
#        numAtoms = sum(pop_list)
#        for n in range(numAtoms + 1): f.readline()
        # get system's center of mass
#        f.readline()
#        totMass = 0
#        cm = zeros(3)
#        for spec, pop in zip(spec_list, pop_list):
#            mass = spec_dict[spec][1]
#            for n in range(pop):
#                totMass += mass
#                coord = array([float(val) for val in f.readline().split()])
#                cm += coord * mass
#
#        cm /= totMass            
## get total charge to make system neutral later
#dv = 1/ng
#dx, dy, dz = 1/ngx, 1/ngy, 1/ngz
#totalChg = 0
#for x in range(ngx):
#    for y in range(ngy):
#	for z in range(ngz):
#	    totalChg += chg3d[z][y][x]
#
#avgChg = totalChg/ng
#print(totalChg, avgChg, dv)
#
#
