## Anthony Yoshimura
## 09/15/16

import os

def getPoscar(POSCAR):
    """
    Returns a list of lists pos_table representing a POSCAR
    POSCAR = POSCAR file (string)
    """
    pos_raw = open(POSCAR)
    pos_str_table = []
    for n in pos_raw:
        pos_str_table.append(n.split())
    
    pos_table = pos_str_table[:]
    for x in range(len(pos_str_table)):
        for y in range(len(pos_str_table[x])):
            a = pos_str_table[x]
            try:
                if a[y].isdigit():
                    b = int(a[y])
                else:
                    b = float(a[y])
                pos_table[x][y] = b
            except ValueError:
                pass

    return pos_table


def write(pos_table, destination = '.'):
    """ 
    exports pos_table as POSCAR_scr in specified directory file
    pos_table = table representing POSCAR (list of lists)
    destination = directory for POSCAR_scr (string)
    """
    try:
        os.makedirs(destination)
    except FileExistsError:
        ##print('directory already exists')
        pass

    f = open(destination + '/POSCAR_scr','w')
    for x in range(len(pos_table)):
        f = open(destination + '/POSCAR_scr','a')
        for y in range(len(pos_table[x])):
            f = open(destination + '/POSCAR_scr','a')
            f.write(' ')
            f.write(str(pos_table[x][y]))
        f.write('\n')


def makeVacancies(POSCAR, vac_list, destination = '.'):
    """
    Makes intial and final POSCARs for vacancy migrations in a TMD lattice
    with specific nearest neighbor vacancies.

    POSCAR = pristine POSCAR file or table (string or list of lists)
    vac_list = list of vacancy positions as shown in VESTA (list)
    """
    if type(POSCAR) == str:
        pos_table = getPoscar(POSCAR)
    elif type(POSCAR) == list:
        pos_table = POSCAR

    ## add 7 (+8: POSCAR header, -1: python counting) to all vacancy positions
    new_vac_list = []
    for n in vac_list:
        n = n + 7
        new_vac_list.append(n)
    vac_list = new_vac_list

    ## recount atom populations
    pop_list = pos_table[6]

    new_pop_list = []
    tot_pop = 7
    for pop in pop_list:
        tot_pop = tot_pop + pop
        for n in vac_list:
            if n > tot_pop - pop and n < tot_pop:
                pop = pop - 1
        new_pop_list.append(pop)

    pos_table[6] = new_pop_list

    vac_list.sort(reverse = True)

    for n in vac_list:
        pos_table.pop(n)

    write(pos_table, destination)


