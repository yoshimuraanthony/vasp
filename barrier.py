# Anthony Yoshiura
# 02/27/17

# extract energies from neb and endpoint OUTCARs
# writes to file 'kinTherm' the kinetic and thermodynamic energy barriers
# all OUTCARs below current directory will be included
# remember to unzip OUTCAR files before running

import matplotlib.pyplot as plt
from plotEnCon import getEnergies

def getBarriers(must_contain = '', write = False, get_end_points = True):
#    system_dict, neb_dict = getEnergies(must_contain = must_contain + '/neb')
    system_dict, neb_dict = getEnergies(must_contain = must_contain)
    if get_end_points:
        init_dict = getEnergies(must_contain = must_contain + '/init')[1]
        fin_dict = getEnergies(must_contain = must_contain + '/fin')[1]

    barrier_table = []
    for system in neb_dict:
        top = max(neb_dict[system])
        if get_end_points:
            init = init_dict[system][0]
            fin = fin_dict[system][0]
        else:
            init = neb_dict[system][0]
            fin = neb_dict[system][-1]

        kin = str(round(top - init, 8))
        therm = str(round(fin - init, 8))
        reverse = str(round(top - fin, 8))

        barrier_table.append([system, kin, reverse, therm])
        
    if write:
        with open('kinTherm', 'w') as f:
            f.write('system\tkinetic\t\treverse\t\tthermo\n')
            for row in barrier_table:
                for val in row:
                    f.write(val + '\t') 
                f.write('\n')

    return barrier_table

def plot(title = 'Energy Profile', system = '', grid = True):
    """
    run in neb output directory containing image directories
    endpoint directories must contain OUTCARs
    """
    system_dict, neb_dict = getEnergies()
    if '00' not in system_dict['.']:
        print('00 OUTCAR not found. Please add OUTCARs to endpoint directories')
        return
    fig = plt.figure(figsize = (8, 5))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize = 20) 
    ax.set_ylabel('Energy (eV)', fontsize = 16) 
    ax.set_xlabel('Image Index', fontsize = 16) 

    if grid:
        ax.grid()

    en_list = neb_dict['.']
    ax.plot(en_list, linewidth = 2)
    ax.plot(en_list, 'bs', markersize = 8)
    ax.set_xlim(0, len(en_list) - 1)
    plt.savefig('profile%s.pdf' %system)
    plt.show()

def plotAll(title = 'Energy Profile', must_contain = ''):
    system_dict, neb_dict = getEnergies(must_contain = must_contain)
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize = 20) 
    ax.set_ylabel('Energy (eV)', fontsize = 16) 
    ax.set_xlabel('Image Index', fontsize = 16) 

    en_list = neb_dict['.']
    ax.plot(en_list, linewidth = 3)
    ax.plot(en_list, 'bs', markersize = 12)
    ax.set_xlim(0, len(en_list) - 1)
    plt.savefig('profile%s.pdf' %system)
    plt.show()


#------------------------------------------------------------------------
# calling from terminal
if __name__ == '__main__':
    getBarriers()
