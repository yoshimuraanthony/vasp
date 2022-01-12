# Cross section and MD paprameters for ion irradiation on TMDs
# Anthony Yoshimura
# 02/06/18

from numpy import pi
from numpy import exp, log, cos, sin, tan, arccos, arcsin, arctan, sqrt, floor, ceil
from numpy import radians, degrees
from numpy import array, arange, linspace, zeros, transpose, dot
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from periodic import table as p_dict
from gaussxw import gaussxwab
from plotPot import getGaussPosEn, getLJ, getPower, getExp, getMDPosEn
from plotEnCon import getEnergy
from scipy.optimize import curve_fit
from os import walk, getcwd

from makeReadable import alignColumns
from getVDAT import xdat2vdat
from getGCD import getGCD
import POSCAR as p

#----------------------------- CONSTANTS -----------------------------------
me = 5.109989461e5 # eV
mp = 9.3827231e8   # eV
c  = 299792458     # m/s
alpha = 1/137      # unitless
electron = 0.085424543065330344 # unitless
joulesPerEV = 1.60217662e-19 # J/eV 
hbar = 1.054571800e-34 # Js

vGa = 2.8710705460453947 # fs/A (velocity of Ga at 30 keV)

# get atomic masses
mGa = p_dict['Ga'][1] * mp # eV
MMo = p_dict['Mo'][1] * mp # eV
MW = p_dict['W'][1] * mp # eV

#----------------------------- SAMPLING POTENTIAL --------------------------

class GaPOSCAR(p.POSCAR):
    """
    subclass of POSCAR with Ga close to target atom
    """
    def addGa(self, b, theta, phi, targAtom = 12):
        """
        adds Ga near the target atom 
        b, theta, phi: spherical coordinates (floats)
        targAtom: target atom number starting from 1 (pos int)
        """
        targCoord = self.coords[targAtom - 1]
        sphCoord = [b, theta, phi]
        dirCoord = self.getDirOf(sphCoord, sourceRep = 'spherical')
        GaCoord = array(dirCoord) + array(targCoord)
        self.add('Ga', GaCoord.tolist())

    def copy(self):
        """ returns identical GaPOSCAR class """
        coordsAndSd = [self.coords[n] + self.sd[n] for n in range(self.getNumAtoms()) ]

        table = [self.name] + [self.scale] + self.cell.tolist() + [self.specs]\
            + [self.pops] + [[self.rep]] + coordsAndSd + [[]] + self.iv

        if self.SD:
            table.insert(7, ['Selective Dynamics'])

        if self.ktable != None:
            ktable = [self.kcomment] + [[self.knumber]] + [[self.ktype]] \
                + [self.kmesh] + [self.kshift]
        else:
            ktable = None 

        return GaPOSCAR(table, ktable)

    def prepPotCalc(self,
            nB = 2,
            nTheta = 3,
            nPhi = 3,
            bRange = [0, 0.02],
            thetaRange = [0, 180],
            phiRange = [90, 150],
            targAtom = 12,
            ):
        """
        prepares POSCAR files for potential energy grid calculation
        """
        b_ar, theta_ar, phi_ar = getSphericalGrid(
                                 nB, nTheta, nPhi, bRange, thetaRange, phiRange)
        for b in b_ar:
            for theta in theta_ar:
                for phi in phi_ar:
                    pos = self.copy()
                    pos.addGa(b, theta, phi, targAtom)
                    outDir = '%.3g_%.3d_%.3d' %(b, theta, phi)
                    outfile = '%s/POSCAR_scr' %outDir
                    pos.write(outfile)

    def prepSimpleRecoilMD(self,
            alpha_list, # radians
            phi_list, # radian
            E = 30000, # eV
            incAtom = 76,
            targAtom = 12,
            d = 0.5, # Å
            ):
        """
        prepares POSCAR files to initialize MD simulation for simplified scattering
            of incident atom and recoil of target atom after collision.  Writes to
            directories labelled by LAB scattering angle and azimuth angle in degrees.
        alpha_list: list of LAB scattering angles in degrees (list of floats)
        phi_list: list of azimuth angles in degrees (list of floats)
        E: incident LAB frame energy in eV (float)
        incAtom: atomic index (starting from 1) of incident atom (pos int)
        targAtom: atomic index (starting from 1) of target atom (pos int)
        d: dist in Å at which trajectory is not affected by interaction (float)
        """
        # get masses and atomic numbers of incident and target atoms
        incSpec = self.getSpecOf(incAtom)
        targSpec = self.getSpecOf(targAtom)
        z = p_dict[incSpec][0] # unitless
        Z = p_dict[targSpec][0] # unitless
        m = p_dict[incSpec][1] * mp # eV
        M = p_dict[targSpec][1] * mp # eV
        
        # calculate center of mass KE
        Ep = M / (m + M) * E # eV
    
        # calculate incident atom's initial velocity
        u = sqrt(2 * E / m) # unitless
        print('u = %s' %u)
    
        # loop through LAB scattering angles
        for alpha in alpha_list:

            # convert to radians
            radAlpha = radians(alpha)

            # calculate CM scattering angle
            alphap = getAlphapFromAlpha(radAlpha, m = m, M = M, showIterations = False) # radians

            # calculate LAB scattering velocity
            v = M * u / (m + M) * sqrt(1 + m**2 / M**2 + 2 * m * cos(alphap) / M) # unitless
            print('v = %s' %v)
    
            # calculate LAB recoil angle
            beta = pi / 2 - alphap / 2 # radians

            # calculate LAB recoil velocity
            V = 2 * m * u * cos(beta) / (m + M) # unitless
    
            # loop through azimuth angles
            for phi in phi_list:

                # convert to radians
                radPhi = radians(phi)

                # position incident atom just after collision
#                psi = arctan(b / d)
#                theta = pi - psi - radAlpha
                theta = pi - radAlpha
                cartRx = d * sin(theta) * cos(radPhi)
                cartRy = d * sin(theta) * sin(radPhi)
                cartRz = d * cos(theta)
                incCoords = array(self.getDirOf([cartRx, cartRy, cartRz]))
                center = array(self.coords[targAtom - 1][:]) # center of spherical coordinates
                incCoords += center

                # calculate cartesian components of v
                cartvx = v * sin(radAlpha) * cos(radPhi)
                cartvy = v * sin(radAlpha) * sin(radPhi)
                cartvz = -v * cos(radAlpha)
                cartv = array([cartvx, cartvy, cartvz]) # unitless
    
                # calculate cartesian components of V
                cartVx = -V * sin(beta) * cos(radPhi)
                cartVy = -V * sin(beta) * sin(radPhi)
                cartVz = -V * cos(beta)
                cartV = array([cartVx, cartVy, cartVz]) # unitless
    
                # convert to Å / fs
                cartv *= c * 1e-5 # Å
                cartV *= c * 1e-5 # Å
                print('cartv = %s' %cartv)
    
                # set initial velocities in POSCAR
                POSCAR_init = self.copy()
                POSCAR_init.setIV(cartv, incAtom)
                POSCAR_init.setIV(cartV, targAtom)
                POSCAR_init.coords[incAtom - 1] = incCoords.tolist()
    
                # write to directory
                dirName = '%.3g_%.3g' %(alpha, phi)
                POSCAR_init.write('%s/POSCAR_scr' %dirName)
    
    def prepRecoilMD(self,
            alpha_list, # radians
            phi_list, # radian
            E = 30000, # eV
            incAtom = 76,
            targAtom = 12,
            d = 0.5, # Å
            ):
        """
        prepares POSCAR files to initialize MD simulation for scattering of incident
            atom and recoil of target atom after collision.  Writes to directories
            labelled by LAB scattering angle and azimuth angle in degrees.
        alpha_list: list of LAB scattering angles in degrees (list of floats)
        phi_list: list of azimuth angles in degrees (list of floats)
        E: incident LAB frame energy in eV (float)
        incAtom: atomic index (starting from 1) of incident atom (pos int)
        targAtom: atomic index (starting from 1) of target atom (pos int)
        d: dist in Å at which trajectory is not affected by interaction (float)
        """
        # UNDER CONSTRUCTION: write in terms of D: distance to next atom
        # get masses and atomic numbers of incident and target atoms
        incSpec = self.getSpecOf(incAtom)
        targSpec = self.getSpecOf(targAtom)
        z = p_dict[incSpec][0] # unitless
        Z = p_dict[targSpec][0] # unitless
        m = p_dict[incSpec][1] * mp # eV
        M = p_dict[targSpec][1] * mp # eV
        
        # get constants (k) from MD runs
        root = '/Users/anthonyyoshimura/Desktop/meunier/ionIrrad/MD/collision/%sS2/280/pc467_333' %targSpec
        XDATCAR = '%s/XDATCAR' %root
        OSZICAR = '%s/OSZICAR' %root
        OUTCAR = '%s/OUTCAR' %root
        R, param_dict = getFitParams(XDATCAR, OSZICAR, OUTCAR, incAtom = incAtom, targAtom = targAtom)
        k = param_dict['a']
        print('coulombic constant: %s' %k)

        # calculate center of mass KE
        Ep = M / (m + M) * E # eV
    
        # calculate incident atom's initial velocity
        u = sqrt(2 * E / m) # unitless
    
        # loop through LAB scattering angles
        for alpha in alpha_list:

            # convert to radians
            radAlpha = radians(alpha)

            # calculate CM scattering angle
            alphap = getAlphapFromAlpha(radAlpha, m = m, M = M, showIterations = False) # radians

            # calculater impact parameter
            b = 2 * Ep / k / tan(alphap / 2) # eV^-1
            b *= 1e10 * hbar * c / joulesPerEV # Å

            # calculate LAB scattering velocity
            v = M * u / (m + M) * sqrt(1 + m**2 / M**2 + 2 * m * cos(alphap) / M) # unitless
    
            # calculate LAB recoil angle
            beta = pi / 2 - alphap / 2 # radians

            # calculate LAB recoil velocity
            V = 2 * m * u * cos(beta) # unitless
    
            # loop through azimuth angles
            for phi in phi_list:

                # convert to radians
                radPhi = radians(phi)

                # position incident atom just after collision
                psi = arctan(b / d)
                R = sqrt(d**2 + b**2) # Å
                theta = pi - psi - radAlpha
                cartRx = R * sin(theta) * cos(radPhi)
                cartRy = R * sin(theta) * sin(radPhi)
                cartRz = R * cos(theta)
                incCoords = array(self.getDirOf([cartRx, cartRy, cartRz]))
                center = array(self.coords[targAtom - 1][:]) # center of spherical coordinates
                incCoords += center

                # calculate cartesian components of v
                cartvx = v * sin(radAlpha) * cos(radPhi)
                cartvy = v * sin(radAlpha) * sin(radPhi)
                cartvz = -v * cos(radAlpha)
                cartv = array([cartvx, cartvy, cartvz]) # unitless
    
                # calculate cartesian components of V
                cartVx = -V * sin(beta) * cos(radPhi)
                cartVy = -V * sin(beta) * sin(radPhi)
                cartVz = -V * cos(beta)
                cartV = array([cartVx, cartVy, cartVz]) # unitless
    
                # convert to Å / fs
                cartv *= c * 1e-5 # Å
                cartV *= c * 1e-5 # Å
    
                # set initial velocities in POSCAR
                POSCAR_init = self.copy()
                POSCAR_init.setIV(cartv, incAtom)
                POSCAR_init.setIV(cartV, targAtom)
                POSCAR_init.coords[incAtom - 1] = incCoords
    
                # write to directory
                dirName = '%.3g_%.3g' %(alpha, phi)
                POSCAR_init.write('%s/POSCAR_scr' %dirName)
        
    
def getSphericalGrid(
        nB = 2,
        nTheta = 3,
        nPhi = 3,
        bRange = [0, 0.1],
        thetaRange = [0, 180],
        phiRange = [90, 150],
        ):
    """
    returns points on spherical grid
    """
    bStart, thetaStart = 0, 0
    thetaEnd, phiEnd = None, None
    minB, maxB = bRange
    if minB == 0:
        nB += 1
        bStart = 1
    b_ar = linspace(minB, maxB, nB)[bStart:]

    minTheta, maxTheta = thetaRange
    if minTheta == 0:
        nTheta += 1
        thetaStart = 1
    if maxTheta == 180:
        nTheta += 1
        thetaEnd = -1
    theta_ar = linspace(minTheta, maxTheta, nTheta)[thetaStart: thetaEnd]

    minPhi, maxPhi = phiRange
    if minPhi == 0 and maxPhi == 360:
        nPhi += 1
        phiEnd = -1
    phi_ar = linspace(minPhi, maxPhi, nPhi)[:phiEnd]

    return b_ar, theta_ar, phi_ar


#--------------------------------- KINETICS ---------------------------------------
def getAlphapFromAlpha(
        alpha,
        alphap = 'auto',
        m = mGa,
        M = MMo,
        cutoff = 1e-10,
        showIterations = True,
        inputUnits = 'radians',
        outputUnits = 'radians',
        ):
    """
    returns CM scattering angle given LAB frame scattering angle
    alpha: LAB scattering angle in radians (float)
        * very unstable near pi/2
    alphap: initial guess of CM scattering angle
    cutoff: precision criterion for Neston's method (float << 1)
    """
    # convert alpha to radians if given in degrees
    if inputUnits[0] == 'd' or inputUnits[0] == 'D':
        alpha = radians(alpha)

    # make initial guess just off the pole
    if alphap == 'auto':
        if alpha < pi / 2:
            alphap = arccos(-m/M) - 1e-3
        else:
            alphap = arccos(-m/M) + 1e-3

    # avoid problems with pole
    if abs(alpha - pi/2) < 1e-10:
        alphap = arccos(-m/M)

    # Newton's method to solve to alphap
    else:
        eps = pi
        while abs(eps) > cutoff:
            eps = getRootFunc(alpha, alphap, m, M) / getRootFuncp(alpha, alphap, m, M)
            alphap -= eps
            if showIterations:
                print('alphap = %s, eps = %s' %(alphap, eps))

    # convert alpha' into degrees if desired
    if outputUnits[0] == 'd' or outputUnits[0] == 'D':
        alphap = degrees(alphap)

    return alphap

def getBfromAlpha(
        alpha,
        E = 30000,
        incSpec = 'Ga',
        targSpec = 'Mo', # 'W'
        k = 203, # eV * Å # 201
        ):
    """
    returns impact paremeter in Å given LAB frame scattering angle
    alpha: LAB scattering angle in radians (float)
        * very unstable near pi/2
    E: LAB frame incident kinetic energy in eV (float)
    k: unitless coulombic constant (float)
    """
    # get masses of incident and target atoms
    m = p_dict[incSpec][1] * mp # eV
    M = p_dict[targSpec][1] * mp # eV

    # get CM frame scattering angle
    alphap = getAlphapFromAlpha(alpha, m = m, M = M, showIterations = False)

    # get CM frame kinetic energy
    Ep = E * M / (m + M) # eV

    # calculate impact parameter
    b = k / 2 / Ep / tan(alphap / 2) # Å
    print('b = %s Å' %b)
    return b

def getRootFunc(alpha, alphap, m, M):
    """
    returns a function whose root is the scattering angle in the CM frame
    alpha: LAB frame scattering angle in radians (float)
    alphap: CM frame scattering angle in radians (float)
    m, M: masses for incident and target atoms espectively in eV (floats)
    """
    return sin(alphap) / (cos(alphap) + m / M) - tan(alpha)
    

def getRootFuncp(alpha, alphap, m, M):
    """
    returns derivative of function whose root is the scattering angle in the CM frame
    alpha: LAB frame scattering angle in radians (float)
    alphap: CM frame scattering angle in radians (float)
    m, M: masses for incident and target atoms espectively in eV (floats)
    """
    return cos(alphap) / (cos(alphap) + m / M) + sin(alphap)**2 / (cos(alphap) + m / M)**2 


def getAlphaFromD(
        alpha,
        alphap = 'auto',
        m = mGa,
        M = MMo,
        cutoff = 1e-10,
        showIteratiosn = True,
        ):
    """
    returns
    """
    pass

def getAlphaFromB(
        b,
        E = 30000,
        incSpec = 'Ga',
        targSpec = 'W',
        degrees = False,
        ):
    """
    returns LAB scattering angle
    b: impact parameter in Angstroms (float)
    """
    # get masses and atomic numbers of incident and target atoms
    z = p_dict[incSpec][0] # unitless
    Z = p_dict[targSpec][0] # unitless
    m = p_dict[incSpec][1] * mp # eV
    M = p_dict[targSpec][1] * mp # eV
    
    # calculate k (columb prefactor) and center of mass KE in natural units
    k = z * Z * electron**2 # unitless
    Ep = M / (m + M) * E # eV

    # convert impact parameter from A to inverse eV
    b /= 1e10 * hbar * c # J^-1
    b *= joulesPerEV # eV^-1
    print(b)
   
    # calculate CM alpha and convert to LAB alpha
    alphap = 2 * arccot(2 * b * E / k)
    print(alphap)
    alpha = sin(alphap) / (cos(alphap) + m / M)

    if degrees:
        alpha = degrees(alpha)

    return alpha


#----------------------------- CROSS SECTION -------------------------------

def getPosEn(top = '.', inNum = 49, targNum = 6):
    """
    returns enery and position arrays from vasp runs
    top: directory that contains vasp run directories (str)
    inNum: index number (a la vesta) of incident atom (pos int)
    targNum: index number (a la vesta) of target atom (pos int)
    """
    # list directories
    dir_list = list(next(walk('.'))[1])
    dir_list.sort()

    # get distances and energies from vasp runs
    dist_list = []
    en_list = []
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
        
            # Get distance between incident and target atoms
            for n in range(targNum):
                f.readline()
            pos1 = array([float(val) for val in f.readline().split()[:3]])
            for n in range(inNum - targNum - 1):
                f.readline()
            pos2 = array([float(val) for val in f.readline().split()[:3]])
            dif = pos1 - pos2
            dif = dot(dif, cell)
            dist = norm(dif)
        dist_list.append(dist)

        # get energies from OUTCARs
        en = getEnergy('%s/OUTCAR' %directory)
        en_list.append(en)

    en_list = (array(en_list) - en_list[-1]).tolist()

    return dist_list, en_list


def getInv(r, a, b): 
    """ 
    returns inverse power type potential at position r
    a: multiplicative constant (pos float)
    b: exponent (pos float)
    """
    return a / r**b


def getFitParams(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        OUTCAR = 'OUTCAR',
        calculationType = 'MD',
        incAtom = 76,
        targAtom = 12,
        ref = 'auto',
        ):
    """
    returns list of fitted parameters for selected fitting function
    XDATCAR: XDATCAR file containing atom positions (str)
    OSZICAR: OSZICAR file containing free energies (str)
    calculationType: type of vasp calculations used to find U ('Norm', 'MD', or 'gauss')
        * norm: NSW = 0 vasp runs are saved in directories labelled by position
        * MD: MD with SMASS = -2 run is saved in current directory
        * guass: same as 'norm', but positions were chosen specially for gaussian quadrature
    incAtom: atom number (starting from 1) whose position is tracked (pos int)
        * if atom = 'auto', atom with most movement is chosen
    ref: reference energy set equal to zero (eV)
        * if 'auto', energy is chosen from static VASP run
    """
    # get distances and energies from specified type of vasp run
    if calculationType[0] == 'm' or calculationType[0] == 'M':
        dist_list, en_list = getMDPosEn(XDATCAR, OSZICAR, OUTCAR, incAtom, targAtom, ref)
    elif calculationType[0] == 'g' or calculationType[0] == 'G':
        dist_list, en_list = getGaussPosEn(incAtom = incAtom, targAtom = targAtom)
    else:
        dist_list, en_list = getPosEn(incAtom = incAtom, targAtom = targAtom)

    # guess initial parameters (eps, rm)
    en_min = min(en_list) - en_list[0]
    r_min = dist_list[en_list.index(min(en_list))]
    guess = (-en_min, r_min, 12, 6)

    # inverse power fit
    print('fitting inverse power potential')
    a, b = curve_fit(getInv, dist_list, en_list, p0 = (60, 2))[0]
    print('exp fitted parameters:')
    print('    a = %.4g, b = %.4g' %(a, b))

    # coefficient of determination for inverse power
    mean = sum(en_list) / len(en_list)
    ss_tot = sum( [(en - mean)**2 for en in en_list] )
    ss_res = sum( [(en - getInv(r, a, b))**2 for en, r in zip(en_list, dist_list)] )
    R = 1 - ss_res / ss_tot
    print('coefficient of determination for inverse fit:\n    R = %.10g' %R)

    # return fitted parameters
    return R, {'a': a, 'b': b}


def getRMin(d = 0,
        inSpec = 'Ga',
        targSpec = 'W',
        Ti = 30000,
        r0 = None,
        eps = None,
        rm = None,
        a = None,
        b = None):
    """
    returns minimum r, the lower bound of the sweeping angle integration
    d: impact parameter in Angstrom (float)
    inSpec: species of incident ion (str)
    targSpec: species of target ion (str)
    Ti: kinetic energy of incident ion eV (float)
    r0: guess for rmin (float)
    """
    # read fitted parameters from vasp runs
    if eps == None:
        R, param_dict = getFitParams()
        eps = param_dict['eps']
        rm = param_dict['rm']
        a = param_dict['a']
        b = param_dict['b']

    # masses
    m = p_dict[inSpec][1]
    M = p_dict[targSpec][1]

    # reduced mass and initial velocity
    mu = m * M / (m + M)
    u = sqrt(2 * Ti / m)

    # relation between potential, impact parameter d, and distance r
    c = 2 / mu / u**2
    if r0 == None:
        r0 = 0.8 * rm  # initial guess just below bond length
    f = c * getExp(r0, eps, rm, a, b) + d**2 / r0**2 - 1
    fp = c * getExpPrime(r0, eps, rm, a, b) - 2 * d**2 / r0**3

    # first recursion to find rmin
    r1 = r0 - f / fp
    prec = r0 / 1000000

    while abs(r1 - r0) > prec:
        print(r0)
        r0 = r1
        r1 = r0 - f / fp
        f = c * getExp(r0, eps, rm, a, b) + d**2 / r0**2 - 1
        fp = c * getExpPrime(r0, eps, rm, a, b) - 2 * d**2 / r0**3

    return r1


def getDTheta(r, U, d, mu, Ti):
    """
    returns integrand of sweeping angle
    r: distance between particles (float)
    U: potential energy (float)
    d: impact parameter (float)
    mu: reduced mass (float)
    Ti: initial kinetic energy in CM frame (float)
    """
    u = sqrt(2*Ti/mu)
    radical = 1 - d**2/r**2 + 2*U/mu/u**2
    return 1/r**2/sqrt(radical)
    

def getTheta(d, u, inSpec = 'Ga', targSpec = 'W', inNum = 49, targNum = 6, top = '.', N = 20):
    """
    reads energies from vasp runs and returns integral of DTheta over sweeping angle
    d: impace parameter (float)
    u: reltive velocity between atoms (float)
    inSpec: species of incident atom (str)
    targSpec: species of target atom (str)
    top: top directory (str)
    N: number of Gaussian quadrature points
    """
    rMin = getRMin(d, inSpec, targSpec, Ti) #put DTheta in terms of Ti, not u
    x_ar, w_ar = gaussxwab(N, rMin, 10) 
    mi = p_dict[inSpec][1]
    mt = p_dict[targSpec][1]
    
#    U_ar = (pe.getEnergies(printPaths = False)[0]['.'])
    param_dict = getFitParams(inNum = 49, targNum = 6, fit = 'exp')[1]
    U_ar = getExp(r, **param_dict)
    I = getDTheta(x_ar, U_ar, d, u) * w_ar
    return sum(I)


#--------------------------------------- MD -----------------------------------------
def prepMD(b_list, phi_list, E = 30000, incSpec = 'Ga', targSpec = 'Mo', infile = 'POSCAR'):
    """
    prepares POSCAR files in directories labelled by the direct coordinates
        of the incident atom
    b_list: list of impact parameters in Angstrom (list of floats)
    phi_list: list of azimuth angles in degrees (list of floats)
    E: incident energy in eV (float)
    incSpec: species of incident ion (str)
    targSpec: species of target atom (str)
    infile: POSCAR file that contains an initial system (str)
    """
    # Create POSCAR instance from infile
    POSCAR_init = p.POSCAR(infile)
    cell = POSCAR_init.cell

    # get incident and target atom numbers
    incAtom, targAtom, minDist = getIncAndTarg(infile)
    targCoord = POSCAR_
    
    for b in b_list:
        for phi in phi_list:
            x = b * sin(phi)
            y = b * cos(phi)
            cartCoords = array([x, y, 0])
            dirCoords = dot(cartCoords, cell)
 
def getIncAndTarg(infile = 'XDATCAR', timeStep = 0.001):
    """
    returns atom numbers (starting from 0) of incident and target atoms
    infile: XDATCAR or POSCAR file (str)
        * assumes incident atom is the highest at t = 0
    """
    if 'XDAT' in infile:
        # create POSCAR instance from first frame in XDATCAR
        POSCAR_init = p.getFromXDAT(1, infile = infile, timeStep = timeStep)
    else:
        # create POSCAR instance from POSCAR file
        POSCAR_init = p.POSCAR(infile)
    POSCAR_init.makeCartesian()

    # find incident atom's horizontal coordinates
    maxHeight = 0
    highestAtom = 0
    refHeight = POSCAR_init.coords[0][2]
    for atom, coord in enumerate(POSCAR_init.coords):
        newHeight = abs(coord[2] - refHeight)
        if newHeight > maxHeight:
            maxHeight = newHeight
            highestAtom = atom
    print('incident atom: %s' %(highestAtom + 1))
    refPosition = array(POSCAR_init.coords[highestAtom][:-1])

    # find target atom (closest to incident atom's horizontal coordinates)
    minDist = norm(POSCAR_init.cell)
    closestAtom = 0
    for atom, coord in enumerate(POSCAR_init.coords):
        newDist = norm(array(coord[:-1]) - refPosition)
        if newDist < minDist and atom != highestAtom:
            minDist = newDist
            closestAtom = atom
    print('target atom: %s' %(closestAtom + 1))
    print('impact parameter: %s' %minDist)

    return highestAtom, closestAtom, minDist


def getRecAngle(infile = 'XDATCAR', targAtom = 'auto'):
    """
    returns recoil angle of target atom
    infile: XDATCAR file (str)
    targAtom: site number of target atom starting from 1 (pos int)
    """
    # get incident and target atom numbers
    if targAtom == 'auto':
        incAtom, targAtom, b = getIncAndTarg(infile = infile)
    else:
        targAtom -= 1
    
    # get final velocity of target atom
    v_tab, num_frames, num_atoms = xdat2vdat(infile = infile, write = False, catch = True)
    traj = v_tab[-1][targAtom]

    # determine angle
    unit = traj / norm(traj)
    recAngle = degrees(arccos(-unit[2]))

    return recAngle


def getScatAngle(infile = 'XDATCAR', incAtom = 'auto'):
    """
    returns scattering angle of target atom
    infile: XDATCAR file (str)
    incAtom: site number of target atom starting from 1 (pos int)
    """
    # get incident and target atom numbers
    if incAtom == 'auto':
        incAtom, targAtom, b = getIncAndTarg(infile = infile)
    else:
        incAtom -= 1
    
    # get final velocity of target atom
    v_tab, num_frames, num_atoms = xdat2vdat(infile = infile, write = False, catch = True)
    traj = v_tab[-1][incAtom]

    # determine angle
    unit = traj / norm(traj)
    scatAngle = degrees(arccos(-unit[2]))

    return scatAngle

#---------------------------------- PLOTTING -------------------------------------
def plotKinEn(
        infile = 'XDATCAR',
        bounds = None,
        timeStep = 0.01,
        title = 'auto',
        save = False,
        outfile = 'auto',
        zero = True,
        note = '',
        ):
    """
    plots kinetic energy of target atom throughout simulation
    bounds: range of frames to be included (list of 2 pos ints)
    atom_list: list of atoms as labelled in VESTA (list of pos ints)
    infile_list: list of XDATCAR files containing atomic positions (str)
    zero: if True, initial kinetic energy is set to zero (bool)
    zeroAtom: reference atom for initial velocity (pos int)
        * if 'auto': reference atom is the first atom in atom_list
    """
    # UNDER CONSTRUCTION: PLOT XDATCARS FROM DIFFERENT DIRECTORIES
    # create POSCAR instance from first frame in XDATCAR
    POSCAR_init = p.getFromXDAT(1, infile = infile, timeStep = timeStep)
    POSCAR_init.makeCartesian()

    # get incident and target atom numbers
    incAtom, targAtom, b = getIncAndTarg(infile = infile)

    # get azimuth angle
    axis = array(POSCAR_init.coords[incAtom][:-1]) - array(POSCAR_init.coords[targAtom][:-1])
    unit = axis / norm(axis)
    phi = degrees(arccos(unit[1]))
    
    # create dictionary with atomic masses
    mass_dict = {}
    i = 0
    for spec, pop in zip(POSCAR_init.specs, POSCAR_init.pops):
        mass = p_dict[spec][1] * mp # mass in eV
        for n in range(pop):
            mass_dict[i] = (spec, mass)
            i += 1

    # get velocities for XDATCAR
    v_tab, num_frames, num_atoms = xdat2vdat(infile = infile, timeStep = timeStep)

    # set initial velocity to zero with respect to target atom
    v_init = [0.0, 0.0, 0.0]
    if zero:
        v_init = v_tab[0][targAtom]
        print('subtracting velocity %s A/fs' %v_init)

    # collect kinetic energies for each atom
    ke_list = [] # list of kinetic energies for target atom
    mass = mass_dict[targAtom][1]
    for frame in range(len(v_tab)):
        v = v_tab[frame][targAtom] - v_init
        s = norm(v) * 1e5 / c # unitless speed
        ke = mass * s**2 / 2
        ke_list.append(ke)
    print('final KE: %s' %ke)

    # plot
    fig, ax = plt.subplots()
    time_list = arange(num_frames) * timeStep

    # label curve by impact parameter and azimuth angle
    label = r'b = %.2g $\AA$, $\phi$ = %.3g$^o$' %(b, phi)
    ax.plot(time_list, ke_list, lw = 2, label = label)

    # create title with chemical formula
    if title == 'auto':
       GCD = getGCD(POSCAR_init.pops[:-1])
       subscripts = [int(pop / GCD) for pop in POSCAR_init.pops[:-1]]
       chemFormula = ''
       for spec, subscript in zip(POSCAR_init.specs[:-1], subscripts):
           if subscript > 1:
               chemFormula += '%s$_%s$' %(spec, subscript)
           else:
               chemFormula += spec
       title = 'KE of %s through collision' %POSCAR_init.specs[0]

    # draw text and formatting
    ax.set_title(title, fontsize = 14)
    ax.set_ylabel('kinetic energy (eV)', fontsize = 12)
    ax.set_xlabel('time (fs)', fontsize = 12)
    if bounds == None:
        beg, end = 0, -1
    else:
        beg, end = bounds
    ax.set_xlim(time_list[beg], time_list[end])
    ax.legend()
    ax.grid()

    # add note
    ax.text(0.99, 0.99, note, fontsize = 12, verticalalignment = 'top', horizontalalignment = 'right',color = 'red', transform = ax.transAxes)

    # label outfile with impact parameter and angle
    if save:
        if outfile == 'auto':
            try:
                suffix = '_%03d_%02d' %(b * 1000, phi)
            except ValueError:
                suffix = '000_00'
            outfile = 'kinEn%s.pdf' %suffix
        plt.savefig(outfile)

    plt.show()


def plotDist(
        infile = 'XDATCAR',
        title = 'auto',
        save = False,
        outfile = 'auto',
        timeStep = 0.01,
        tbounds = None,
        dbounds = None,
        note = 'auto',
        incAtom = 'auto',
        targAtom = 'auto',
        ):
    """
    plots distance between incident and target atoms thoughout simulation
    tbounds: range of frames to be included (list of 2 pos ints)
    dbounds: range of distances to be included (list of 2 pos ints)
    infile: XDATCAR file (str)
    """
    # create POSCAR instance from first frame in XDATCAR
    POSCAR_init = p.getFromXDAT(1, infile = infile, timeStep = timeStep)
    POSCAR_init.makeCartesian()

    # get incident and target atom numbers
    if incAtom == 'auto' and targAtom == 'auto':
        incAtom, targAtom, b = getIncAndTarg(infile = infile)

    dif = abs(incAtom - targAtom)

    if incAtom < targAtom:
        atom1 = incAtom
        atom2 = targAtom
    else:
        atom1 = targAtom
        atom2 = incAtom

    with open(infile) as f:
        for n in range(2): # skip to cell
            f.readline()
        cell = [ [float(val) for val in f.readline().split()] for n in range(3)]
        latLengths = [norm(vec) for vec in cell]
        f.readline() # skip to pops
        pops = [int(val) for val in f.readline().split()]
        num_atoms = sum(pops)
 
        # skip to "Direct configuration 1"
        dist_list = []
        i = 1
        for line in f:
            i += 1
            for n in range(atom1):
                f.readline()
            coord1 = array([float(val) for val in f.readline().split()])

            for n in range(dif - 1):
                f.readline()
            coord2 = array([float(val) for val in f.readline().split()])
            delta = dot(coord1 - coord2, cell)
            for n in range(3):
                delta[n] -= round(delta[n] / latLengths[n]) * latLengths[n]
            dist = norm(delta)
            dist_list.append(dist)

            for n in range(num_atoms - atom2 - 1):
                f.readline()
            
    minDist =  min(dist_list)

    # plotting
    fig, ax = plt.subplots()
    time_list = arange(i - 1) * timeStep
    print(len(time_list), len(dist_list))
    ax.plot(time_list, dist_list, lw = 2) 
    title = 'interatomic distance through collision'
    ax.set_title(title, fontsize = 14)
    ax.set_xlabel('time (fs)', fontsize = 12)
    ax.set_ylabel(r'distance ($\AA$)', fontsize = 12)
    if tbounds == None:
        beg, end = 0, -1
    else:
        beg, end = tbounds
    ax.set_xlim(time_list[beg], time_list[end])
    if dbounds == None:
        bot, top = 0, max(dist_list[beg: end]) * 1.1
    else:
        bot, top = dbounds
    ax.set_ylim(bot, top)
    ax.grid()
    ax.text(.01, .99, 
        r"impact parameter = %.5g $\AA$" %b, verticalalignment = 'top',
        transform = ax.transAxes, fontsize = 12) 
    ax.text(.01, .92, 
        r"minimum distance = %.5g $\AA$" %minDist, verticalalignment = 'top',
        transform = ax.transAxes, fontsize = 12) 

    # add note
    if note == 'auto':
        note = getcwd().split('/')[-1]
    ax.text(0.99, 0.99, note, fontsize = 12, verticalalignment = 'top', horizontalalignment = 'right',color = 'red', transform = ax.transAxes)

            
    if save:
        if outfile == 'auto':
            try:
                suffix = '_%03d' %(b * 1000)
            except ValueError:
                suffix = '000_00'
            outfile = 'dist%s.pdf' %suffix
            
        plt.savefig(outfile)

    plt.show()


#----------------------------------- HELPERS --------------------------------------
cell_dict = {
    'MoS2': array([[15.94442892, 0, 0],[-7.97221446, 13.80828049, 0],[0, 0, 20]]),
    'WS2' : array([[15.9456507, 0, 0],[-7.97282535, 13.80933859, 0],[0, 0, 20]])
    }

def arccot(ratio):
    return pi / 2 - arctan(ratio)


def getPhiAndB(dirDisp = [-1/30, 1/30, 0], tmd = 'WS2'):
    """
    returns phi (degrees) and b (A) given horizontal displacement in direct coordinates
    dirDisp: horizontal displacement in direct coordinates (list of two floats)
    tmd: tmd species ('MoS2' or 'WS2')
    """
    while len(dirDisp) < 3:
        dirDisp = list(dirDisp)
        dirDisp.append(0)

    cell = cell_dict[tmd]
    cartDisp = dot(dirDisp, cell)
    b = norm(cartDisp)
    unit = cartDisp / b
    phi = degrees(arcsin(unit[0]))

    return phi, b

def getDirDisp(phi = 0, b = 0.1, tmd = 'WS2'):
    """
    returns horizontal displacement in direct coordinates given b (A) and phi (degrees)
    phi: azimuth angle (degrees) with respect to Sulfur bond (float)
    b: impact parameter (A) (float)
    tmd: tmd species ('MoS2' or 'WS2')
    """
    cell = cell_dict[tmd]
    inv_cell = inv(cell)
    cartDisp = array([-b * sin(phi), b * cos(phi), 0])
    
    return dot(cartDisp, inv_cell)
    

def getPhiAndBFromXDAT(infile = 'XDATCAR'):
    """
    returns phi (degrees) and b (Å) given horizontal displacement in direct coordinates
    dirDisp: horizontal displacement in direct coordinates (list of two floats)
    tmd: tmd species ('MoS2' or 'WS2')
    """
    # get incident and target atom numbers
    incAtom, targAtom, b = getIncAndTarg(infile = infile)

    # get POSCAR from first frame of XDATCAR
    POSCAR_init = p.getFromXDAT(1, infile)

    # get displacement and tmd species
    dirDisp = array(POSCAR_init.coords[incAtom]) - array(POSCAR_init.coords[targAtom])
    dirDisp[-1] = 0
    specs = POSCAR_init.specs
    if 'W' in specs:
        tmd = 'WS2'
    elif 'Mo' in specs:
        tmd = 'MoS2'
           
    return getPhiAndB(dirDisp, tmd)


def getRefEns():
    """
    returns dictionary of reference energies for getMDPosEn
    """
    pass

#----------------------------------- SCRATCH ---------------------------------------
def getExpPrimePrime(r, eps, rm, a, b):
    """
    Returns second derivative of exponential Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    {a,b}: exponents on first and second terms (pos ints)
    """
    c = eps * a * b / (a - b) / rm**2
    return c * (a * exp(a*(1 - r / rm)) - (b + 1) * (r / rm)**(b + 2))


def getTmax(Te, M, m = me):
    """
    returns maximum energy (eV) transfer from elastic collision
    species: atom being displaced (str)
    Te: kinetic energy of incident electron in eV (float)
    m: mass of incident particle in eV (float)
    """
    if M > 1000 * m:
        return 2*Te*(Te + 2*m) / M   # m << M
    else:
        return 4*M*m*Te / (M + m)**2 # M ~ m, Te << mc^2


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


def getPowerPrime(r, eps, rm, a, b):
    """
    Returns derivative of Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    {a,b}: exponents on first and second terms (pos ints)
    """
    c = eps * a * b / (a - b) / rm
    return c * ((rm / r)**(b + 1) - (rm / r)**(a + 1))


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


def getExpPrime(r, eps, rm, a, b):
    """
    Returns derivative of exponential Lennard-Jones type potential at position r
    eps: minimum energy (float)
    rm: position of energy minimum (float)
    {a,b}: exponents on first and second terms (pos ints)
    """
    c = eps * a * b / (a - b) / rm
    return c * ((rm / r)**(b + 1) - exp(a*(1 - r/rm)))


def oldPlotFit(
        XDATCAR = 'XDATCAR',
        OSZICAR = 'OSZICAR',
        inNum = 49,     # energy with respect to interatomic distance
        targNum = 6,
        title = 'auto',
        save = False,
        outfile = 'lj.pdf',
        grid = True,
        ):
    """
    plots energy vs position data with curve fit
    XDATCAR: XDATCAR file containing atom positions (str)
    OSZICAR: OSZICAR file containing free energies (str)
    inSpec: atom number (starting from 1) whose position is tracked (pos int)
        * if inSpec = 'auto', atom with most movement is chosen
    """
    # get interatomic distances and energies
    dist_list, en_list = getPosEn(inNum = inNum, targNum = targNum)

    # guess initial parameters (eps, rm)
    en_min = min(en_list) - en_list[0]
    r_min = dist_list[en_list.index(min(en_list))]
    guess = (-en_min, r_min, 12, 6)

    # Lennard-Jones-power fit
    print('fitting Lennard-Jones-power potential')
    peps, prm, pa, pb = curve_fit(getPower, dist_list, en_list, p0 = guess)[0]
    print('power fitted parameters:')
    print('    epsilon = %.4g eV, rm = %.4g A, a = %.4g, b = %.4g' %(peps, prm, pa, pb))

    # Lennard-Jones-exponential fit
    print('fitting Lennard-Jones-exp potential')
    eeps, erm, ea, eb = curve_fit(getExp, dist_list, en_list, p0 = guess)[0]
    print('exp fitted parameters:')
    print('    epsilon = %.4g eV, rm = %.4g A, a = %.4g, b = %.4g' %(eeps, erm, ea, eb))

    # coefficient of determination for power
    mean = sum(en_list) / len(en_list)
    ss_tot = sum( [(en - mean)**2 for en in en_list] )
    ss_res = sum( [(en - getPower(r, peps, prm, pa, pb))**2 for en, r in zip(en_list, dist_list)] )
    PR = 1 - ss_res / ss_tot
    print('coefficient of determination for power fit:\n    R = %.10g' %PR)

    # coefficient of determination for exp
    mean = sum(en_list) / len(en_list)
    ss_tot = sum( [(en - mean)**2 for en in en_list] )
    ss_res = sum( [(en - getExp(r, eeps, erm, ea, eb))**2 for en, r in zip(en_list, dist_list)] )
    ER = 1 - ss_res / ss_tot
    print('coefficient of determination for exp fit:\n    R = %.10g' %ER)

    # array of positions (plotting domain)
    lower = min(dist_list) - .01
    upper = max(dist_list) + .01
    r = linspace(lower, upper, 200)

    # fitted curve
    Power = getPower(r, peps, prm, pa, pb)
    Exp = getExp(r, eeps, erm , ea, eb)

    # prepare figure
    fig, ax = plt.subplots()

    # plot data and fit
    ax.plot(dist_list, en_list, 'ro', label = 'Data', zorder = 3)
    ax.plot(r, Power, label = 'Pow: R = %.4g' %PR, zorder = 3)
    ax.plot(r, Exp, label = 'Exp: R = %.4g' %ER, zorder = 3)

    # title
    if title == 'auto':
        title = 'Ga - W interaction potential'

    # plot formatting
    ax.set_xlim(lower, upper)
    ax.set_ylim(min(en_list) - 1, max(en_list) + 1)
    ax.set_title(title, fontsize = 14)
    ax.set_xlabel(r'position ($\AA$)', fontsize = 12)
    ax.set_ylabel('energy (eV)', fontsize = 12)
    ax.legend(loc = (.66, .67))
    ax.axhline(y = 0, color = 'k', ls = '--', zorder = 2)
    if grid:
        ax.grid()

    # display fitted parameters
    ax.text(.99, .99, r'$\epsilon$ = %.3g eV, $r_m$ = %.3g $\AA$, $a$ = %.3g, $b$ = %.3g'
          %(peps, prm, pa, pb),
           fontsize = 12, ha = 'right', va = 'top', transform = ax.transAxes, color = 'tab:blue')
    ax.text(.99, .92, r'$\epsilon$ = %.3g eV, $r_m$ = %.3g $\AA$, $a$ = %.3g, $b$ = %.3g'
          %(eeps, erm, ea, eb),
           fontsize = 12, ha = 'right', va = 'top', transform = ax.transAxes, color = 'tab:orange')

    if save:
        plt.savefig(outfile)
    plt.show()

def getvFromAlphap(alphap, m = mGa, M = MMo, E = 30000):
    u = sqrt(2 * E / m) # unitless
    v = M * u / (m + M) * sqrt(1 + m**2 / M**2 + 2 * m * cos(alphap) / M) # unitless
    return v # unitless


def getSecondEFromAlphap(alphap, m = mGa, M = MMo, E = 30000):
    u = getvFromAlphap(alphap, m, M, E) # unitless
    V = 2 * m * u / (m + M) # unitless
    return M * V ** 2 / 2 # eV

#    # Lennard-Jones-power fit
#    if 'pow' in fit or 'Pow' in fit:
#        print('fitting Lennard-Jones-power potential')
#        eps, rm, a, b = curve_fit(getPower, dist_list, en_list, p0 = guess)[0]
#        print('power fitted parameters:')
#        print('    epsilon = %.4g eV, rm = %.4g A, a = %.4g, b = %.4g' %(eps, rm, a, b))
#
#        # coefficient of determination for power
#        mean = sum(en_list) / len(en_list)
#        ss_tot = sum([(en - mean)**2 for en in en_list])
#        ss_res = sum([(en - getPower(r, eps, rm, a, b))**2 for en, r in zip(en_list, dist_list)])
#        R = 1 - ss_res / ss_tot
#        print('coefficient of determination for power fit:\n    R = %.10g' %R)
#
#    # Lennard-Jones-exponential fit
#    if 'exp' in fit or 'Exp' in fit:
#        print('fitting Lennard-Jones-exp potential')
#        eps, rm, a, b = curve_fit(getExp, dist_list, en_list, p0 = guess)[0]
#        print('exp fitted parameters:')
#        print('    epsilon = %.4g V, rm = %.4g A, a = %.4g, b = %.4g' %(eps, rm, a, b))
#
#        # coefficient of determination for xp
#        mean = sum(en_list) / len(en_list)
#        ss_tot = sum( [(en - mean)**2 for en in en_list] )
#        ss_res = sum( [(en - getExp(r, eps, rm, a, b))**2 for en, r in zip(en_list, dist_list)] )
#        R = 1 - ss_res / ss_tot
#        print('coefficient of determination for exp fit:\n    R = %.10g' %R)
#
#    return R, {'eps': eps, 'rm': rm, 'a': a, 'b': b}
#    # ensure alphap is between 0 and pi
#    alphap -= 2 * pi * round(alphap / 2 / pi) # bring between -pi and pi
#    if alphap < 0:
#        alphap += 2 * pi
#        print(alphap)

            # calculate LAB scattering angle
#            alpha = arctan(sin(alphap) / (cos(alphap) + m / M))
#            alphap = 2 * arccot(2 * b * Ep / k)
#        b_list: list of impact parameters in Angstroms (list of floats)
        # calculate k (columb prefactor) and center of mass KE in natural units
#        k = z * Z * electron**2 # unitless

#    b *= 1e10 * hbar * c / joulesPerEV # Å
