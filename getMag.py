# Anthony Yoshimura
# 11/07/2017

from makeReadable import alignDecimal
from numpy.linalg import norm

def getMag(infile = 'OUTCAR', write = False, outfile = 'MAGMOM'):
    """
    returns table containing magnetic moments for each ion
    infile: OUTCAR file from non-collinear calculation (str)
    write:  if True, writes magnetic moments in outfile (bool)
    outfile: name of written file containing magnetic moments (str)
    """
    with open(infile) as f:
        for line in f:
            if 'NIONS' in line:
                nions = int(line.split()[-1])

            elif 'magnetization (x)' in line:
                for n in range(3):
                    f.readline()
                x_list = [float(f.readline().split()[-1]) for n in range(nions)]
                
            elif 'magnetization (y)' in line:
                for n in range(3):
                    f.readline()
                y_list = [float(f.readline().split()[-1]) for n in range(nions)]
    
            elif 'magnetization (z)' in line:
                for n in range(3):
                    f.readline()
                z_list = [float(f.readline().split()[-1]) for n in range(nions)]

            elif 'number of electron ' in line:
                tot_list = [float(val) for val in line.split()[-3:]]

    norm_list = [norm(tup) for tup in zip(x_list, y_list, z_list)]
    tot_list.append(norm(tot_list))

    if write:
        # for pasting into INCAR
        with open(outfile, 'w') as f:
            f.write('MAGMOM = ')
            for n in range(nions):
                f.write('%s %s %s ' %(x_list[n], y_list[n], z_list[n]))

        # for human eyes
        with open('POSCAR') as f:
            pos_list = f.readlines()
            spec_list = pos_list[5].split()
            pop_list = [int(val) for val in pos_list[6].split()]

        ax_list = alignDecimal(x_list, 3)
        ay_list = alignDecimal(y_list, 3)
        az_list = alignDecimal(z_list, 3)
        anorm_list = alignDecimal(norm_list, 3)
        atot_list = alignDecimal(tot_list, 7)

        with open('%s_h' %outfile, 'w') as f:
            f.write('magnetic moments integrated around each atomic site\n')
            f.write('\tx\ty\tz\tnorm\n')
            site = 0
            for spec, pop in zip(spec_list, pop_list):
                for n in range(pop):
                    f.write('%.3d %s\t%s\t%s\t%s\t %s\n'
                          %(site + 1, spec,
                            ax_list[site], ay_list[site], az_list[site], anorm_list[site]))
                    site += 1

            f.write('\ntot:')
            for tot in atot_list:
                f.write('\t%s' %tot)
            f.write('\n')

    return x_list, y_list, z_list


def makeSuperMag(mag_tab, dim, infile = 'POSCAR', write = False, outfile = 'MAGMOM'):
    """
    returns table containing magnetic moments for each ion in supercell
    mag_tab: table of mag moments for each ion in x, y, and z directions (list of lists)
    dim: dimensions of supercell
    infile: OUTCAR file from non-collinear calculation (str)
    write:  if True, writes magnetic moments in outfile (bool)
    outfile: name of written file containing magnetic moments (str)
    """
    # avoid aliasing
    mag_tab = [line[:] for line in mag_tab]

    with open(infile) as f:
        for n in range(2):
            f.readline()
        cell = [[float(val) for val in f.readline().split()] for n in range(3)]
        f.readline()
        pops = [int(val) for val in f.readline().split()]

    super_tab = [[] for n in range(3)]
    scale = dim[0] * dim[1] * dim[2]
    last = 0
    for pop in pops:
        first = last
        last += pop
        for n in range(3):
            spec_list = mag_tab[n][first: last]
            super_tab[n] += spec_list * scale

    if write:
        with open(outfile, 'w') as f:
            for n in range(len(super_tab[0])):
                f.write('%s %s %s    ' %(super_tab[0][n], super_tab[1][n], super_tab[2][n]))

    return super_tab

#-------------------------------------------------------------------------------
# calling from terminal
if __name__ == '__main__':
    getMag(write = True)

