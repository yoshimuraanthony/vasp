#!/usr/bin/env python

from numpy import transpose
from numpy.linalg import norm
from os import listdir

def writeForces(outfile = 'forces', infile = 'OUTCAR'):
    """ 
    returns maximum and average forces
    """
    maxForce_list = []
    avgForce_list = []
 
    with open(infile) as f:
        for line in f:  
            if 'NION' in line:
                nions = int(line.split()[-1])

            if 'TOTAL' in line:
                force_list = []
                next(f)
                for ion in range(nions):
                    force_str = next(f).split()[-3:]
                    force = norm([float(val) for val in force_str])
                    force_list.append(force)

                maxForce = max(force_list)
                avgForce = sum(force_list) / nions

#                print('%.3d\t%6.4f\t%6.4g' %(ion + 1, maxForce, avgForce))

                maxForce_list.append(max(force_list))
                avgForce_list.append(sum(force_list) / nions)
    
    nsteps = len(maxForce_list)
    with open(outfile, 'w') as f:
        f.write('iter\tmax\tavg\n')
        for n, maxForce, avgForce in zip(range(nsteps), maxForce_list,
                avgForce_list):
            f.write('%3s\t%6.4f\t%6.4f\n' %(n + 1, maxForce, avgForce))

#--------------------------- calling from terminal ----------------------------
if __name__ == '__main__':
   writeForces()

