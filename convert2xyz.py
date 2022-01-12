# Anthony Yoshimura
# 04/17/17

from numpy import asarray, array, dot, zeros, float64
from time import time
import os

def xdat2xyz(
        infile = 'XDATCAR',
        outfile = 'movieXDAT.xyz',
        start = 0,
        end = None,
        ):
    """
    converts XDATCAR to xyz movie for VMD
    infile: XDATCAR file (str)
    outfile: xyz output file (str)
    start: starting ionic step (pos int)
    end: ending ionic step (pos int)
        * if None, reads to end of infile
    """
    startTime = time()

    with open(infile) as f:
        f.readline()     # skip comment line
        scale = float(f.readline())
        cell_a2 = array([[float(val) for val in f.readline().split()]
                for n in range(3)])
        specs = f.readline().split()
        pops = [int(val) for val in f.readline().split()]
        nions = sum(pops)

        xdat_l3 = []
        for line in f:
            xdat_l2 = []
            for n in range(nions):
                xdat_l2.append([float(val) for val in
                    f.readline().split()])

            xdat_l3.append(xdat_l2)
    xdat_a3 = array(xdat_l3)
    cart_a3 = dot(xdat_a3, cell_a2)

    spec_list = []
    for spec_index in range(len(specs)):
        spec_list += [specs[spec_index] for n in range(pops[spec_index])]

    print('read time = {} seconds'.format(time() - startTime))
    midTime = time()

    with open(outfile, 'w') as f:
        for frame, cart_a2 in enumerate(cart_a3):
            f.write(str(nions))
            f.write('\n')
            f.write(' ')
            f.write('frame {}'.format(frame + 1))
            f.write('\n')
            for ion, cart_ar in enumerate(cart_a2):
                f.write('  ')
                f.write(spec_list[ion].ljust(2))
                for cart in cart_ar:
                    f.write(f'{cart:12.7f}')
                f.write('\n')

    print('write time = {} seconds'.format(time() - midTime))
    print('total time = {} seconds'.format(time() - startTime))



def pos2xyz(infile = 'POSCAR', must_contain = None, outfile = 'moviePOS.xyz'):
    """ converts set of POSCARs to xyz movie for VMD """
    path_list = ['%s/%s' %(f, infile) for f in os.listdir('.') if f[:2].isdigit()]
    cart_t3 = []
    spec_tab = []
    for path in path_list:
        with open(path) as pos:
            pos_table = pos.readlines()
            scale = float(pos_table[1])
            cell = [ [float(val) for val in line.split()] for line in pos_table[2:5]]

            specs = pos_table[5].split()
            pops = [int(val) for val in pos_table[6].split()]
            num_atoms = sum(pops)

            if 'elective' in pos_table[7]:
                start = 9    # skip extra line for selective dynamics
            else:
                start = 8

            dir_list = pos_table[start: num_atoms + start]

        dir_tab = [ [float(val) for val in line.split()[:3]] for line in dir_list]
        cart_tab = [dot(dir_coord, cell) for dir_coord in dir_tab]
        cart_t3.append(cart_tab)

        spec_list = []
        for spec_index in range(len(specs)):
            spec_list += [specs[spec_index] for n in range(pops[spec_index])]
        spec_tab.append(spec_list)

    with open(outfile, 'w') as f:
        for frame in range(len(cart_t3)):
            f.write(str(num_atoms))
            f.write('\n')
            f.write(' ')
            f.write('frame %s' %(frame + 1))
            f.write('\n')
            for atom in range(num_atoms):
                f.write('  ')
                f.write(spec_tab[frame][atom])
                f.write('\t')
                for comp in range(3):
                    f.write(str(cart_t3[frame][atom][comp]))
                    f.write('\t')
                f.write('\n')

#------------------------------------------------------------------------------

# calling from terminal
if __name__ == '__main__':
    xdat2xyz()


def oldXdat2xyz(infile = 'XDATCAR', outfile = 'movieXDAT.xyz', skip = 0):
    """ converts XDATCAR to xyz movie for VMD """
    startTime = time()

    with open(infile) as xdat:
        xdat.readline()     # skip comment line
        scale = float(xdat.readline())
        cell = [ [float(val) for val in xdat.readline().split()] for n in range(3)]
        specs = xdat.readline().split()
        pops = [int(val) for val in xdat.readline().split()]
        num_atoms = sum(pops)
        
        xdat_list = xdat.readlines()

    num_frames = int(len(xdat_list) / (num_atoms + 1))

    xdat_list = [line for line in xdat_list if 'configuration' not in line] # keep only coordinates

    xdat_tab = [ [float(val) for val in line.split()] for line in xdat_list]
    cart_tab = [dot(dir_coord, cell) for dir_coord in xdat_tab]
    spec_list = []
    for spec_index in range(len(specs)):
        spec_list += [specs[spec_index] for n in range(pops[spec_index])]

    print('read time = {} seconds'.format(time() - startTime))
    midTime = time()

    with open(outfile, 'w') as f:
        for frame in range(num_frames):
            f.write(str(num_atoms))
            f.write('\n')
            f.write(' ')
            f.write('frame %s' %(frame + 1))
            f.write('\n')
            for atom in range(num_atoms):
                f.write('  ')
                f.write(spec_list[atom])
                f.write('\t')
                for comp in range(3):
                    f.write(str(cart_tab[frame * num_atoms + atom][comp]))
                    f.write('\t')
                f.write('\n')
    print('write time = {} seconds'.format(time() - midTime))
    print('total time = {} seconds'.format(time() - startTime))

