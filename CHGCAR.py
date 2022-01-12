from numpy import array, reshape, linspace
from copy import deepcopy
from subprocess import call
import POSCAR as p

def writeSuperCHGCAR(
        dim = [2, 2, 2],
        infile = 'CHGCAR',
        outfile = 'CHGCAR_scr',
        VESTA = False,
        ):
    """
    writes CHGCAR file for a supercell of desired dimensions
    infile: CHGCAR file (str)
    dim: super cell dimensions (list of 3 pos ints)
    outfile: file to which super charge density is written (str)
    VESTA: if True, outfile is immediatly plotted in VESTA
    """
    # table to produce POSCAR instance (POSCAR class has makeSuperCell function)
    pos_tab = []

    # read CHGCAR file
    print('reading %s' %infile)
    with open(infile) as f:

        # add POSCAR-related lines to POSCAR
        for line in f:

            # stop adding to POSCAR at first blank line
            if len(line.strip()) > 0:

                # convert strings to ints and floats where needed
                line_list = line.split()
                new_list = []
                for val in line_list:
                    try:
                        if val.isdigit():
                            new_list.append(int(val))
                        else:
                            new_list.append(float(val))
                    except ValueError:
                        new_list = line_list

            else:
                break

            pos_tab.append(new_list)

        # obtain grid dimensions
        ngx, ngy, ngz = [int(val) for val in f.readline().split()]
        ng = ngx * ngy * ngz

        # obtain charge densities as a string
        chg_str = f.read()

    # reshape charge density into 3D grid
    print('reshaping charge density')
    chg_list = chg_str.split()[:ng]
    chg_ar = array( [float(val) for val in chg_list] )
    chg_ar3 = reshape(chg_ar, (ngz, ngy, ngx))
    chg_list3 = chg_ar3.tolist()

    # create super charge density grid
    xdim, ydim, zdim = dim
    factor = xdim*ydim*zdim  # scale up density to cancel expansion
    newZ_list3 = []
    for y_list2 in chg_list3:
        newY_list2 = []
        for x_list in y_list2:
            newY_list2.append(x_list * xdim)
            
        newZ_list3.append(newY_list2 * ydim)

    newChg_list3 = newZ_list3 * zdim

    # get super grid dimenstions
    newNgx, newNgy, newNgz = ngx * xdim, ngy * ydim, ngz * zdim
    newNg = newNgx * newNgy * newNgz

    # convert grid to Nrow x 5col format
    newChg_ar = reshape(newChg_list3, newNg)
    mod = newNg % 5
    numRows = (newNg - mod) // 5
    if mod != 0: 
        newChg_ar2 = reshape(newChg_ar[:-mod], (numRows, 5)) * factor
        newChgTail_ar = newChg_ar[-mod:]
    else:
        newChg_ar2 = reshape(newChg_ar, (numRows, 5))
        newChgTail_ar = []
            
    # create super cell POSCAR
    supercell = p.POSCAR(pos_tab)
    supercell.makeSuperCell(dim)

    # write super CHGCAR
    print('writing to %s' %outfile)
    with open(outfile, 'w') as f:

        # cell and atomic coordinates
        table = [supercell.name] + [supercell.scale] + supercell.cell.tolist() \
              + [supercell.specs] + [supercell.pops] + [[supercell.rep]] \
              + supercell.coords

        for row in table:
            for element in row: 
                f.write(str(element) + ' ') 
            f.write('\n')

        # ngx, ngy, and ngz
        f.write('\n')
        for ngi in [newNgx, newNgy, newNgz]:
            f.write(' ' + str(ngi))
        f.write('\n')           
 
        # charge density
        for row in newChg_ar2:
            for density in row:
                f.write(str(density) + ' ')
            f.write('\n')

        # tail end charge density
        for density in newChgTail_ar:
            f.write(str(density) + ' ')

    # open outfile in VESTA
    if VESTA:
        call("open -a VESTA %s" %outfile, shell = True)

#------------------------------------- NOTES -----------------------------------
# How to plot charge density difference in VESTA

# Edit -> Edit Data -> Volumetric Data
# Import -> browse to CHGCAR file and select "Open"
# Select "Subtract from current data" --> OK

#------------------------------------- SCRATCH -----------------------------------

def getSuperGrid(grid = [[[1,2],[3,4]],[[5,6],[7,8]]], dim = [2,2,2]):
    """
    returns super grid
    grid: 3d list (list of lists of lists)
    dim: super cell dimensions
    """
    xdim, ydim, zdim = dim
    newChg_list3 = deepcopy(grid)
    newChg_list3 = []
    for y_list2 in grid:
        newY_list2 = []
        for z_list in y_list2:
            newY_list2.append(z_list * zdim)
            
        newChg_list3.append(newY_list2 * ydim)

    newChg_list3 *= xdim

    return newChg_list3
