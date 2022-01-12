# Site- and layer-projected DOS plots
# Anthony Yoshimura
# 11/21/16

# Current directory should contain DOSCAR

from numpy import transpose, array, zeros
import matplotlib.pyplot as plt
from plotBS import getBandGap
from copy import deepcopy

# UNDER CONSTRUCTION:
#    fix multiplication to multiply DOSCAR2 in DosFig

            
def getPDOS(DOSCAR = 'DOSCAR', site_list = None, orb_list = None, sorb_list = None):
    """
    Returns dictionary with the following {key: value} pairs

        dos_tab  : table containing total DOS
        bounds   : [emin, emax]
        efermi   : fermi energy
        site     : site
        orbital  : orbital
        
    DOSCAR: DOSCAR file (str)
    site: projection site, as labelled in VESTA (int)
    orbital: 's', 'p', 'd', or 'f'
    """
    with open(DOSCAR) as f:
        DOSCAR_list = f.readlines()

    orb_dict = {'s': 1, 'p': 2, 'd': 3, 'f': 4}
    if type(orb_list) == list:
        orb_list = [orb_dict[orb] for orb in orb_list]
    

    num_sites = int( DOSCAR_list[0].split()[0] )

    header = DOSCAR_list[5].split()               # header values
    emax, emin, nedos, efermi = [float(val) for val in header[:4] ]
    nedos = int(nedos)

    print('number of sites:', num_sites)
    print('fermi level at', efermi)

    # total DOS (contains integrated an en_list)
    dos_tab = []
    for i in range(6, nedos + 6):
        dos_str = DOSCAR_list[i].split()
        dos = [float(val) for val in dos_str]
        dos_tab.append(dos)

    dos_tab = transpose(dos_tab) # tranpose to make plottable in pyplot

    en_list = dos_tab[0]

    # site projected (sum of sites)
    if type(site_list) == list:
        sdos_tab  = zeros(nedos)
        for site in site_list:
            sdos_list = []
            start  = 6 + site * (nedos + 1)
            finish = 5 + (site + 1) * (nedos + 1)

            for j in range(start, finish):
                sdos_str = DOSCAR_list[j].split()
                sdos = array( [float(val) for val in sdos_str[1:]] ).sum()
                sdos_list.append(sdos)
 
            sdos_tab += array(sdos_list)

    else:
        sdos_tab = None

    # orbital projected
    if type(orb_list) == list:
        odos_tab  = []
        for orb in orb_list:
            odos_list = [0 for n in range(nedos)]
            start = 6
            end   = 7 + nedos
            for i in range(num_sites):                # sum orbital pdos for each site
                start += nedos + 1
                end   += nedos + 1
                for j in range(start, end):
                    odos_str = DOSCAR_list.split()
                    odos = float(pdos_str[orb])
                    odos_list[j] += pdos 
       
        odos_tab.append(odos_list)

    else:
        odos_tab = None

    # site and orbital projected
    if type(orb_list) == list and type(site_list) == list:
        sodos_tab = []
        for orb in orb_list:
            sdos_tab = []
            for site in site_list:
                sdos_list = []
                start = 6 + site * (nedos + 1)
                end   = 6 + (site + 1) * (nedos + 1)

                for j in range(start, end):
                    sodos_str = DOSCAR_list[j].split()
                    sodos = float(pdos_str[orb])
                    sodos_list.append(pdos)
       
                sdos_tab += array(sodos_list)

            sodos_tab.append(sdos_tab)

    else:
        sodos_tab = None 

    return {'dos_tab' : dos_tab,    'bounds'  : [emin, emax],  'efermi'   : efermi,
            'sdos_tab': sdos_tab,   'odos_tab': odos_tab,      'sodos_tab': sodos_tab,
            'nedos'   : nedos,      'en_list' : en_list,       'num_sites': num_sites}


#----------------------------------- PLOTTING -----------------------------------------
def plot(
        # parameters
        bounds = None,
        site_list = None,
        orb_list = None,
        maxDen = None,
        subtractFermi = True,
        shift = 0,

        # input files
        DOSCAR = 'DOSCAR',
        DOSCAR2 = None,

        # plot attributes
        title = 'Density of States',
        showFermi = False,
        showFermi2 = False,
        show = True,
        show2 = True,
        showEdges2 = False,
        showLegend = False,
        color = 'grey',
        label = 'total',
        label2 = 'pristine',

        # saving
        save = False,
        outfile = 'dos.pdf',
        ):
    """
    Plots pdos for the specified sites and orbtitals
        * LORBIT

    bounds: energy range to be plotted (list of two floats)
    DOSCAR: DOSCAR file (str)
    DOSCAR2: DOSCAR (with same nedos) file to be compared to DOSCAR (str)
    site_list: list of atom sites (list of ints)
    orb_list: list of atom orbitals (list of letters)
    maxDen: maximum density to be plotted (float) 
    subtractFermi: if True, fermi energy is subtracted from all energies (bool)
    """
    dos_dict = getPDOS(DOSCAR, site_list, orb_list)

    # density of states
    dos_tab   = dos_dict['dos_tab']
    sdos_tab  = dos_dict['sdos_tab']
    odos_tab  = dos_dict['odos_tab']
    sodos_tab = dos_dict['sodos_tab']

    # energy domain
    en_list = array(dos_dict['en_list']) + shift
    efermi  = dos_dict['efermi']
    

    if subtractFermi:
        en_list = (en_list - efermi).tolist()
        emin, emax = array(dos_dict['bounds']) - efermi
    else:
        emin, emax = dos_dict['bounds']
        emin += shift
        emax -= shift

    # other values
    num_sites = dos_dict['num_sites']
    nedos     = dos_dict['nedos']

    # determine indice bounds from energy bounds 
    if type(bounds) == list or type(bounds) == tuple:
        if bounds[0] < emin:
            print('Lower bound less than emin. Setting lower bound to emin')
            bounds[0] = emin
        if bounds[1] > emax:
            print('Upper bound greater than emax. Setting upper bound to emax')
            bounds[1] = emax

    else:
        bounds = [emin, emax]

    lindex = min(range(nedos), key = lambda i: abs(en_list[i] - bounds[0]))
    uindex = min(range(nedos), key = lambda i: abs(en_list[i] - bounds[1]))
   
    # prepare figure with single plot
    fig, ax1 = plt.subplots()

    # total DOS
    ax1.plot(en_list[lindex: uindex], dos_tab[1][lindex: uindex],
                label = label, color = color, linewidth = 2)

    # site projected
    if type(site_list) == list:
        site_string = ''
        for site in site_list:
            site_string += ' ' + str(site)

        #ax1.set_title('Site-projected DOS for sites' + site_string, fontsize = 20)
        ax1.set_title('Site-projected DOS', fontsize = 20)
        file_str = ''.join([str(n) for n in site_list])

    else:
#        ax1.set_title(title, fontsize = 20)
        ax1.text(.02, .98, title, ha = 'left', va = 'top',
                 fontsize = 16, transform = ax1.transAxes)

    if type(site_list) == list:
        ax1.plot(en_list[lindex: uindex], sdos_tab[lindex: uindex], linewidth = 2,
                    #label = 'sites%s' %site_string)
                    label = 'site-projected')

    # orbital projected
    if type(orb_list) == list:
        for i in range(len(orb_list)):
            ax1.plot(en_list[lindex: uindex], odos_tab[i][lindex: uindex],
                        label = 'orbital %s' %orb_list[i])

    if showFermi:
        ax1.annotate('trefoil E$_f$', color = 'tab:red', 
                    fontsize = 12, ha = 'center', va = 'bottom', rotation = 'vertical',
                    xy = (efermi, 0), xycoords = 'data',
                    xytext = (0, 40), textcoords = 'offset points',
                    arrowprops = dict(facecolor = 'tab:red', shrink = 0.05))

    # site and orbital projected (in progress)

    # second DOSCAR file
    if type(DOSCAR2) == str:
        dos2_dict = getPDOS(DOSCAR2, site_list, orb_list)

        dos2_tab   = dos2_dict['dos_tab']
        sdos2_tab  = dos2_dict['sdos_tab']
        odos2_tab  = dos2_dict['odos_tab']
        sodos2_tab = dos2_dict['sodos_tab']
        num2_sites = dos2_dict['num_sites']
        efermi2    = dos2_dict['efermi']
        en2_list = array(dos_dict['en_list'])

        factor = num_sites / num2_sites # account for different number of atoms
        dos2_tab = factor * array(dos2_tab)

        if subtractFermi:
            en_list = (array(en_list) + efermi - efermi2).tolist()

        if show2:
            ax1.plot(en2_list[lindex: uindex], dos2_tab[1][lindex: uindex],
                label = label2, linewidth = 2, zorder = 2, color = 'red')

        if showEdges2:
            edges_dict = getBandGap('EIGENVAL_p', 'POSCAR_p', 'DOSCAR_p')
            vbm, cbm = edges_dict['vbm'], edges_dict['cbm']
            ax1.axvline(x = vbm)
            ax1.axvline(x = cbm)

            # text
            width = bounds[1] - bounds[0]
            vbm_position = (vbm - bounds[0])/width
            cbm_position = (cbm - bounds[0])/width
            ax1.text(vbm_position + .01, .99, "Host\nVBM", va = 'top',
                transform = ax1.transAxes, fontsize = 14)
            ax1.text(cbm_position - .01, .99, "Host\nCBM", ha = 'right', va = 'top',
                transform = ax1.transAxes, fontsize = 14)

            ax1.fill_between(en_list, 0, dos_tab[1], where = en_list <= vbm, color = 'gray')

        if showFermi2:
#            ax1.annotate('pristine E$_f$',
#                        fontsize = 12, ha = 'center', va = 'bottom', rotation = 'vertical',
#                        xy = (efermi2, 0), xycoords = 'data',
#                        xytext = (0, 40), textcoords = 'offset points',
#                        arrowprops = dict(facecolor = 'k', shrink = 0.05))
            ax1.axvline(x = efermi2, ls = '--')
            efermi_position = (efermi2 - bounds[0]) / width
            ax1.text(efermi_position, 0.55, 'pristine E$_f$', color = 'tab:blue',
                    transform = ax1.transAxes, fontsize = 14,
                    ha = 'right', va = 'center', rotation = 'vertical')

    # remaining plot settings
    if showLegend:
        ax1.legend(loc = 1)

    ax1.grid()
    ax1.set_xlim(bounds)

    if maxDen != None:
        ax1.set_ylim(0, maxDen)

    if subtractFermi:
        ax1.set_xlabel('E-E$_f$ (eV)')
    else:
        ax1.set_xlabel('Energy (eV)', fontsize = 16)

    ax1.set_ylabel('Density of States', fontsize = 16)
    plt.tight_layout()

    # save figure
    if save:
#        if outfile != '':
#            if type(site_list) == list:
#                file_str = ''.join([str(n) for n in site_list])
        #        plt.savefig('sites_' + file_str + '.pdf', format = 'pdf')
#                plt.savefig('site_proj.pdf', format = 'pdf')
#            elif type(DOSCAR2) == str:
#                plt.savefig('dos_comp.pdf')
#            else:
#                plt.savefig('dos.pdf')
#        else:
#            plt.savefig(outfile)
        plt.savefig(outfile)

    if show:
        plt.show()


#------------------------------- DOS FIGURE CLASS --------------------------------
class DosFig(object):
    """
    Class designed to hold and plot multiple DOS data sets
    """
    def __init__(self, data_tab = None, nrows = 2, ncols = 2):
        """
        initializes DosFig instance with data_tab
        """
        self.nrows = nrows
        self.ncols = ncols

        # holder for data dicts
        if data_tab == None:
            self.data_tab = [[[] for n in range(ncols)] for m in range(nrows)]
        else:
            self.data_tab = deepcopy(data_tab)

    def __add__(self, other):
        """
        other: object to be added to d_ar's (DosFig, list, array, float, or int)
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n]['d_ar']

                    if type(other) == StrainFig:
                        otherE_ar = other.data_tab[row][col][n]['d_ar']
                    elif type(other) == list or type(other) == ndarray:
                        otherE_ar = other[row][col]
                    elif type(other) == float or type(other) == int:
                        otherE_ar = other

                    selfE_ar += otherE_ar

    def __sub__(self, other):
        """
        other: object to be subtracted from d_ar's (DosFig, list, array, float, or int)
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n]['d_ar']

                    if type(other) == StrainFig:
                        otherE_ar = other.data_tab[row][col][n]['d_ar']
                    elif type(other) == list:
                        otherE_ar = other[row][col]
                    elif type(other) == float or type(other) == int:
                        otherE_ar = other

                    selfE_ar -= otherE_ar

    def __mul__(self, other):
        """
        other: object by which d_ar's are multiplied (DosFig, list, array, float, or int)
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n]['d_ar']
                    selfE_ar *= other

    def __truediv__(self, other):
        """
        other: object by which d_ar's are divided (DosFig, list, array, float, or int)
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n]['d_ar']
                    selfE_ar /= other

    def addData(self,
            DOSCAR = 'DOSCAR',
            DOSCAR2 = 'DOSCAR_p',
            EIGENVAL2 = 'EIGENVAL_p',
            POSCAR2 = 'POSCAR_p',
            label = None,
            row = 0,
            col = 0,
            ):
        """
        adds data to data_tab 
        DOSCAR: DOSCAR file (str)
        """
        e_list = []
        d_list = []
        with open(DOSCAR) as f:
            num_sites = int(f.readline().split()[0])
            for n in range(4):
                f.readline()
            emax, emin, nedos, efermi = [float(val) for val in f.readline().split()[:4]]
            nedos = int(nedos)
    
            print('number of sites:', num_sites)
            print('fermi level at', efermi)
    
            # total DOS 
            for i in range(nedos):
                en, dos, idos = [float(val) for val in f.readline().split()]
                e_list.append(en)
                d_list.append(dos)

        
        edges_dict = getBandGap(EIGENVAL2, POSCAR2, DOSCAR2)
        vbm, cbm, efermi2 = edges_dict['vbm'], edges_dict['cbm'], edges_dict['efermi']

        self.data_tab[row][col].append({
                'e_ar': array(e_list),
                'd_ar': array(d_list),
                'vbm': vbm,
                'cbm': cbm,
                'efermi': efermi,
                'efermi2': efermi2,
                'label': label,
                'nedos': nedos,
                'num_sites': num_sites
                })

    def plot(self, maxDen = 300, figsize = (8, 6), bounds = [-3, .5]):
        """
        plots pdos for the specified sites and orbtitals
        bounds: energy range to be plotted (list of two floats)
        maxDen: maximum density to be plotted (float) 
        """
        self.fig, self.ax_tab = plt.subplots(figsize = figsize, nrows = self.nrows,
                                                                ncols = self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):

                # extract data from dict (e_ar, d_ar, label, etc...)
                data_dict = self.data_tab[row][col][0]
                locals().update(data_dict)

                e_ar = data_dict['e_ar']
                d_ar = data_dict['d_ar']
                vbm  = data_dict['vbm']
                cbm  = data_dict['cbm']
                efermi  = data_dict['efermi']
                efermi2  = data_dict['efermi2']
                label = data_dict['label']
                nedos = data_dict['nedos']
                num_sites = data_dict['num_sites']
                emin, emax = e_ar[0], e_ar[-1]
            
                # prepare axes
                ax = self.ax_tab[row][col]
            
                # plot total DOS
                ax.plot(e_ar, d_ar, color = 'black', linewidth = 2)
            
                # energy bounds 
                if type(bounds) == list:
                    if bounds[0] < emin:
                        print('Lower bound outside of energy range.\
                               Setting lower bound to %s' %emin)
                    else:
                        emin = bounds[0]

                    if bounds[1] > emax:
                        print('Upper bound outside of energy range. Setting upper bound to %s' %emax)
                    else:
                        emax = bounds[1]            
            
                # band edges
                ax.axvline(x = vbm)
                ax.axvline(x = cbm)
            
                width = emax - emin
                vbm_position = (vbm - emin)/width
                cbm_position = (cbm - emin)/width
                ax.text(vbm_position - .01, .99, "Host\nVBM", ha = 'right', va = 'top',
                    transform = ax.transAxes, fontsize = 11)
                ax.text(cbm_position - .01, .99, "Host\nCBM", ha = 'right', va = 'top',
                    transform = ax.transAxes, fontsize = 11)
    
                ax.fill_between(e_ar, 0, d_ar, where = e_ar <= vbm, color = 'gray')
            
                # fermi level with red arrow
                ax.annotate('trefoil E$_f$', color = 'tab:red', 
                            fontsize = 11, ha = 'center', va = 'bottom', rotation = 'vertical',
                            xy = (efermi, 0), xycoords = 'data',
                            xytext = (0, 40), textcoords = 'offset points',
                            arrowprops = dict(facecolor = 'tab:red', shrink = 0.05))

                # pristine fermi level with dashed line
                ax.axvline(x = efermi2, ls = '--')
                efermi_position = (efermi2 - emin) / width
                ax.text(efermi_position, .99, 'pristine E$_f$', color = 'tab:blue',
                        transform = ax.transAxes, fontsize = 11,
                        ha = 'right', va = 'top', rotation = 'vertical')

                # only show x and y labels on left and bottom plots
                if row == self.nrows - 1:
                    ax.set_xlabel('Energy (eV)', fontsize = 11)
                if col == 0:
                    ax.set_ylabel('Density of States', fontsize = 11)
                
            
                # remaining plot settings
                ax.grid()
                ax.set_xlim(emin, emax)
                ax.set_ylim(0, maxDen)
                ax.tick_params(labelsize = 9)

    def setTitle(self, title, row = 0, col = 0, loc = 2):
        """
        add title to specified axes
        """
        ax = self.ax_tab[row][col]

        if loc == 1:
            x, y = .98, .99
            ha, va = 'right', 'top'
        elif loc == 2:
            x, y = .01, .99
            ha, va = 'left', 'top'
        elif loc == 4:
            x, y = .98, .01
            ha, va = 'right', 'bottom'

        ax.text(x, y, title, transform = ax.transAxes,
                ha = ha, va = va, fontsize = 13)

    def copy(self):
        return DosFig(self.data_tab, self.nrows, self.ncols)
       
    def save(self, outfile = 'dosTable.pdf'):
        self.fig.tight_layout()
        self.fig.savefig(outfile)

    def show(self):
        self.fig.tight_layout()
        self.fig.show()
            

#------------------------------- HELPER FUNCTIONS -----------------------------
def getBandGap(DOSCAR = 'DOSCAR', thres = 1e-6):
    """
    gets band gap from DOSCAR
    DOSCAR: DOSCAR file (str)
    thres: minimum density at which energy can be considered occupied
    """
    # UNDER CONSTRUCTION: assumes only one gap
    # extract data from DOSCAR
    with open(DOSCAR) as f:

        num_sites = int(f.readline().split()[0])

        for n in range(4):
            f.readline()

        emax, emin, nedos, efermi = [float(val) for val in f.readline().split()[:4]]
        nedos = int(nedos)
        print('number of sites:', num_sites)
#        print('fermi level at', efermi)

        # get total DOS data
        e_list = []
        d_list = []
        for i in range(nedos):
            en, dos, idos = [float(val) for val in f.readline().split()]
            e_list.append(en)
            d_list.append(dos)

        # find gap in energies
        biggestGap = 0
        inGap = False
        for n in range(len(e_list) - 1):
            nextEn = e_list[n + 1]
            nextDen = d_list[n + 1]

            # find vbm
            if nextDen < thres:
                if inGap == False:
                    vbm = nextEn
                    print('vbm: %s' %vbm)
                    inGap = True

            # find cbm
            else:
                if inGap == True:
                    cbm = e_list[n]
                    print('cbm: %s' %cbm)
                    inGap = False
                    gap = cbm - vbm - .016

                    # subtract small amount from gap to match gap
                    # calculated from bs.getBandGap
                    print('gap = %s' %gap)
                    if gap > biggestGap:
                        biggestGap = gap

#        return {'vbm': vbm, 'cbm': cbm, 'gap': cbm - vbm - .016}
        return {'vbm': vbm, 'cbm': cbm, 'gap': biggestGap}


#-------------------------------------------------------------------------------
# calling from terminal
if __name__ == '__main__':
   plot()
