## Anthony Yoshimura
## 10/22/16

import numpy as np
from POSCAR import getPOSCAR

def getRecip(POSCAR):
    """
    Produces file reciproc.in, containing repricocal vectors as rows,
        in current directory
    POSCAR: POSCAR file (str)
    """
    pos_tab = getPOSCAR(POSCAR)
    
    real_cell = np.array([vec[:] for vec in pos_tab[2:5]])

    rec_cell = np.zeros([3, 3])

    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3 
        cross = np.cross(real_cell[j], real_cell[k])
        volume = np.dot(real_cell[i], cross)
        rec_cell[i] = 2 * np.pi * cross / volume

    with open('reciproc.in', 'w') as f:
        for vec in rec_cell:
            for element in vec:
                f.write(str(element) + ' ')
            f.write('\n')
