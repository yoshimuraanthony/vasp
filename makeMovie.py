# Anthony Yoshimura
# 10/05/18

from os import listdir
from copy import deepcopy
import POSCAR as p

def makeMovieFromNEB():
    """
    makes movie from POSCARs in directories labelled by frame number
        * assumes cell does not change
    """
    # list of current directories
    d_list = [d for d in listdir() if d.isdigit()]
    d_list.sort()

    # get coordinates from all POSCARs
    coords_t3 = []
    for d in d_list:
        pos = p.POSCAR('%s/POSCAR' %d)
        pos.makeCartesian()
        coords = deepcopy(pos.coords)
        coords_t3.append(coords)
       
    # get species, populations, and number of atoms
    range_dict = pos.getSpecRanges()
    num_atoms = pos.getNumAtoms()

    # write movie in to xyz file
    with open('movie_scr.xyz', 'w') as f:
        for n in range(len(coords_t3)):
            f.write('%s\nframe %02d' %(num_atoms, n))
            for spec in range_dict:
                init, fin = range_dict[spec]
                for coord in coords_t3[n][init: fin]:
                    f.write('\n %s' %spec)
                    for val in coord:     
                        f.write('\t%s' %val)
            f.write('\n')

#--------------------------- calling from terminal ----------------------------
if __name__ == '__main__':
    makeMovieFromNEB()
