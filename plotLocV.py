# plot local potential and/or charge density along certain direciton
# Anthony Yoshimura
# 11/21/16

# requires LOCPOT or CHGCAR

from numpy import array, reshape, linspace, polyfit
import matplotlib.pyplot as plt

def plotLPZ(
        LOCPOT = 'LOCPOT',
        OUTCAR = 'OUTCAR',
        fit = False,
        subtractFermi = True,
        title = None,
        save = True,
        outfile = 'lpz.pdf',
        ):
    """
    plots average local potential with respect to z for unit cell
    LOCPOT: LOCPOT file (str)
    fit: if True, fits lines to vacuum portions (useful to find E-fields) (bool)
    """
    with open(LOCPOT) as inputFile_LOCPOT:

        for n in range(4): inputFile_LOCPOT.readline()
    
        height = float( inputFile_LOCPOT.readline().split()[-1] )
        
        inputFile_LOCPOT.readline()
    
        pop_str = inputFile_LOCPOT.readline().split()
        pop = array([int(val) for val in pop_str]).sum()
        
        for n in range(pop + 2): inputFile_LOCPOT.readline()
    
        ng_str = inputFile_LOCPOT.readline().split()
        ngx, ngy, ngz = [int(val) for val in ng_str]
        ng = ngx * ngy * ngz
    
        pot_str = inputFile_LOCPOT.read()
        pot_list = pot_str.split()[:ng]
        pot_1d = array( [float(val) for val in pot_list] )
    
        pot_3d = reshape(pot_1d, (ngz, ngy, ngx))
    
        pot_z = [layer.mean() for layer in pot_3d]

    with open(OUTCAR) as inputFile_OUTCAR:
        for line in inputFile_OUTCAR:
            if 'E-fermi' in line:
                efermi = float(line.split()[2])

    print('Fermi energy = %s' %efermi)

    z = linspace(0, height, ngz)
    vac_pot = max(pot_z)
    work_func = vac_pot - efermi
    vac_pot_index = pot_z.index(vac_pot)
    vac_pot_z = z[vac_pot_index]
    
    print('vacuum potential = %s' %vac_pot)
    print('z position = %s' %vac_pot_z)

#    m, b = polyfit(z[10:30], pot_z[10:30], 1)
#    fit1 = m*z + b
#    lower2, upper2 = int( (height - 3)*10 ), int( (height - 1)*10 )
#    m2, b2 = polyfit(z[lower2: upper2], pot_z[lower2: upper2], 1)
#    fit2 = m2*z + b2

    fig, ax = plt.subplots()
    ax.plot(z, array(pot_z) - efermi)
    ax.set_xlim(0, height)
    if fit:
        ax.plot(z, fit1, linestyle = 'dashed')
        ax.text(2.5, 5.5, r'slope $\simeq$ %s (eV/$\AA$)' %round(m, 2), fontsize = 16, color = 'g')
        ax.plot(z, fit2, linestyle = 'dashed')
        ax.text(.2, -3.5, r'slope $\simeq$ %s (eV/$\AA$)' %round(m2, 2), fontsize = 16, color = 'r')

#    ax.set_title(r'Average local potential core-excited MoSe$_2$', fontsize = 20)
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel(r'$z$ $(\AA)$', fontsize = 16)
    ax.set_ylabel(r'$V - E_F$ (eV)', fontsize = 16)
    
    ax.annotate(r'$\Phi$ = %.3g eV' %work_func,
            fontsize = 14, ha = 'center',
            xy = (vac_pot_z, work_func), xycoords = 'data',
            xytext = (0, -40), textcoords = 'offset points',
            arrowprops = dict(facecolor = 'k', shrink = 0.05))

    ax.grid()
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    plt.show()

