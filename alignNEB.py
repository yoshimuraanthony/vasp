# Anthony Yoshimura
# 09/27/17

import POSCAR as p
import os
from subprocess import call

def align(infile = 'CONTCAR', top = '.'):
    """
    writes POSCARs into NEB directories with equal CMs
    run in nebsave directory
    """
    path_list = [f for f in os.listdir() if f.isdigit()]

    start_path = '%s/POSCAR' %path_list[0]
    end_path = '%s/POSCAR' %path_list[-1]
    start_pos = p.POSCAR(start_path)
    start_pos.write(start_path)
    start_cm = start_pos.getCM()

    for path in path_list:
        try:
            pos = p.POSCAR('%s/%s' %(path, infile))
        except FileNotFoundError:
            pos = p.POSCAR('%s/POSCAR' %path)
        pos_cm = pos.getCM()
        tran = start_cm - pos_cm
        pos.translate(tran)
        pos.write('%s/POSCAR_al' %path)

#-------------------------------------------------------------------------------
# calling from terminal
if __name__ == '__main__':
    align()
