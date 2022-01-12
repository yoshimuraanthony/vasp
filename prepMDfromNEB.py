from numpy import array, dot, sqrt
from numpy.linalg import norm
from copy import deepcopy
from periodic import table as p_dict
import POSCAR as p

#kb = 8.6173303e-5 #eV/K
kb = 1.38064852e-23 #J/K 
mp = 1.6726219e-27  #kg  


def getMDPOSCAR(T = 933, uniform = False):
    """
    returns instance of POSCAR class for MD simulation based on initial NEB displacement
        ** assumes first two NEB image directories are named '00' and '01'
    T: temperature in K (pos float)
    uniform: if True, initial velocites are uniform, corresponding to 3/2kT (bool)
    """
    pos_0 = p.POSCAR('00/POSCAR')
    pos_1 = p.POSCAR('01/POSCAR')
    pos_0.makeCartesian()
    pos_1.makeCartesian()

    # get the masses for all atoms in POSCARs
    specRanges_dict = pos_0.getSpecRanges()
    m_list = []
    for spec in specRanges_dict:
        m = p_dict[spec][1] * mp
        min, max = specRanges_dict[spec]
        for n in range(min, max):
            m_list.append(m)
            
    # get the displacement between POSCARs 0 and 1
    x0_ar = array(deepcopy(pos_0.coords))
    x1_ar = array(deepcopy(pos_1.coords))

    if uniform:
        dx_ar = array([(x1 - x0) / norm(x1 - x0) for x0, x1 in zip(x0_ar, x1_ar)])
    else:
        dx_ar = (x1_ar - x0_ar) # Angstrom

#    dxSquared_list = [dot(dx, dx) / norm(dx)**2 for dx in dx_ar]
    dxSquared_list = [dot(dx, dx) for dx in dx_ar]

    # get something proporational to the total kinetic energy
    ETimesConstant = sum(m * dxSquared for m, dxSquared in zip(m_list, dxSquared_list))
        
    # get the initial velocities for each atom as a function of T
    numAtoms = pos_0.getNumAtoms()

    C = sqrt(3 * numAtoms * kb * T / ETimesConstant)

    v_list = (C * dx_ar * 1e-5).tolist() # A/fs

    # create POSCAR with calculated initial velocities
    posMD = pos_0.copy()
    posMD.setIV(v_list, atom_list = 'all')
    
    return posMD




