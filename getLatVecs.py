from numpy import array, cos, sin, radians, sqrt

def getLatVecs(a, b, c, alpha, beta, gamma):
    """
    returns lattice vectors in POSCAR format
    a, b, c: lattice parameters (Å)
    alpha: angle between b, c vectors (degrees)
    beta: angle between a, c vectors (degrees)
    gamme: angle between a, b vectors (degrees)
    """
    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)

    return array(
            [[a, 0, 0], [b*cos(gamma), b*sin(gamma), 0],\
            [c*cos(beta), c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma),\
            c*sqrt(1 - cos(beta)**2\
            - ((cos(alpha) - cos(beta)*cos(gamma))/sin(gamma))**2)]])

    

