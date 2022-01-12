from numpy import array
import os 

def writeEnergies(
        col_list = [0, 10, 20, 30, 40, 50],
        outfile1 = 'en.txt',
        outfile2 = 'enOcc.txt',
        ):
    """
    writes energies as txt file
    """
    en_a2, occ_a2 = readOUTCARs(col_list)
    nb, nd = en_a2.shape

    with open(outfile1, 'w') as f:
        for b, en_ar in enumerate(en_a2):
            f.write('{:>3}'.format(nb-b))
            for en in en_ar:
                f.write('{:>10.4f}'.format(en))
            f.write('\n')                

    with open(outfile2, 'w') as f:
        for b, (en_ar, occ_ar) in enumerate(zip(en_a2, occ_a2)):
            f.write('{:>3}'.format(nb-b))
            for en, occ in zip(en_ar, occ_ar):
                f.write('{:>12.4f}{:>9.5f}'.format(en, occ))
            f.write('\n')                


def readOUTCARs(col_list=[0,10,20,30,40,50]):
    """
    returns arrays of energies from all */OUTCAR
    col_list: column indices to read
    """
    d_list = next(os.walk('.'))[1]
    d_list.sort()

    en_l2 = []
    occ_l2 = []
    for col in col_list:
        d = d_list[col]
        with open('{}/OUTCAR'.format(d)) as f:
            for line in f:
                if 'NBANDS' in line:
                    nb = int(line.split()[-1])

                if 'band energies' in line:
                    en_list = []
                    occ_list = []
                    for b in range(nb):
                        _, en, occ = [float(val) for val in
                                f.readline().split()]
                        en_list.append(en)
                        occ_list.append(occ)

                    en_l2.append(en_list)
                    occ_l2.append(occ_list)

    # transpose and reverse order of bands
    return array(en_l2).T[::-1], array(occ_l2).T[::-1]

