# Anthony Yoshimura
# 05/03/18

from getGCD import getGCD
import POSCAR as p

def getChemForm(infile = 'POSCAR'):
    """
    returns string representing chemical formula in LaTeX format
    infile: POSCAR or XDATCAR file
    """
    # UNDER CONSTRUCTION: get from OUTCAR
    # ions per type = 
    # VRHFIN =

    # get specs and pops from POSCAR or XDATCAR
    if 'POS' in infile or 'pos' in infile:
        POSCAR = p.POSCAR(infile)
    elif 'XDAT' in infile or 'xdat' in infile:
        POSCAR = p.getFromXDAT(1, infile) 

    specs, pops = POSCAR.specs, POSCAR.pops 

    # divide pops by greatest common denominator to obtain chemical subscripts
    GCD = getGCD(pops)
    subscripts = [int(pop / GCD) for pop in pops]
    chemFormula = ''
    for spec, subscript in zip(specs, subscripts):
        if subscript > 1:
            chemFormula += '%s$_{%s}$' %(spec, subscript)
        else:
            chemFormula += spec

    return chemFormula

  

