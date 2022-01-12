## Anthony Yoshimura
## 09/15/16

## intended for pristine TMD monolayer

from vacancy import *
import itertools as it

sd = {199: True, 201: True, 218: True, 219: True, 180: True, 181: True}

def swap(POSCAR, pair):
    """
    swaps the atom indices for a pair of atoms
    POSCAR = pristine POSCAR file or table (string or list of lists)
    pair = pair of atoms indices as shown in VESTA (list)
    """
    if type(POSCAR) == str:
        pos_table = getPoscar(POSCAR)
    if type(POSCAR) == list:
        pos_table = POSCAR

    ## add 7 (+8: POSCAR header, -1: python counting) to all vacancy positions
    pair[0] += 7
    pair[1] += 7

    swap_table = pos_table[:]
    swap_table[pair[0]] = pos_table[pair[1]]
    swap_table[pair[1]] = pos_table[pair[0]]

    return swap_table


def makeMigration(POSCAR, site_dict = sd):
    """
    creates initial and final POSCARs for a migration in a specified
    configuration of nearest neighbor vacancies
    POSCAR = pristine POSCAR file (string)
    site_dict = indicates if site is occupied (dictionary)
    """
    ## make destination
    top_bot = '/'
    left_right = ''

    possible_vacs = [199, 201, 218, 219, 180, 181]

    for site in possible_vacs[2:]:
        if not site_dict[site]:
            top_bot = top_bot + '0'
        else:
            top_bot = top_bot + '1'

    for site in possible_vacs[:2]:
        if not site_dict[site]:
            left_right = left_right + '0'
        else:
            left_right = left_right + '1'

    ## make list for makeVancacies
    vac_list = []
    for site in site_dict:
        if not site_dict[site]:
            vac_list.append(site)

    vac_list = [200] + vac_list

    POSCAR_swap = swap(POSCAR, [198,200])

    makeVacancies(POSCAR, vac_list, left_right + top_bot + '/init')
    makeVacancies(POSCAR_swap, vac_list, left_right + top_bot + '/fin')
    

def makeAllMigrations(POSCAR):
    """
    creates all unique migration POSCARs and saves them to the corresponding directory
    POSCAR = pristine POSCAR file (string)
    """
    for bool_list in it.product([True, False], repeat = 6):
        possible_vacs = [199, 201, 218, 219, 180, 181]
        site_dict = {}
        for n in range(6):
            site_dict[possible_vacs[n]] = bool_list[n]

        makeMigration(POSCAR, site_dict)
