# Anth ny Yoshimura
# 11/09/17

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import listdir
from plotEnCon import getEnergy, getVolume, getPaths
from scipy.optimize import curve_fit
from numpy import linspace, array, floor
from copy import deepcopy

# labels and directories
label_dict = {'x': 'zigzag', 'y': 'armchair', 'u': 'biaxial'}
TMD_tab = [['MoS2', 'WS2'], ['MoSe2', 'WSe2']]
top = '/Users/anthonyyoshimura/Desktop/meunier/trefoil'
edim_tab = [[-7.14875291, -7.14875291], [-5.96423809, -5.96423809]]
epris_tab = [[-24.37128030, -26.12497703], [-22.54108906, -24.04401997]]

# plotting keyward arguments
splitKwargs = dict(color = 'k', clip_on = False, zorder = 3, linewidth = 1)
tickKwargs = dict(left = True, right = True, direction = 'in')

# diagonal line segments indicating broken axes
d = 0.02
x1, y1 = (-d, d), (-d + 1.02, d + 1.02)
x2, y2 = (1 - d, 1 + d), (0.98 - d, 0.98 + d) 


def plotTMDFig(save = False, outfile = 'strain.pdf', figsize = (8, 8)):
    """
    plots panel of strain plots with broken axes
    """
    # prepare table of axes
    fig, ax_tab = plt.subplots(nrows = 4, ncols = 2,
                               sharex = True, figsize = figsize)
    fig.subplots_adjust(hspace = 0.0, wspace = 0.0)

    # holders for trefoil energy data under uniform strain
    top_tab = [[] for n in range(4)]
    bot_tab = [[] for n in range(4)]

    # main loop (count backwards to obtain epris before calculating formation energies)
    for row in range(4):
        y_list = []
        for col in range(2):

            # get axes and values from tables
            chal = int(floor(row/2))
            epris = epris_tab[chal][col] * 81
            edim = edim_tab[chal][col]
            tmd = TMD_tab[chal][col]
            ax = ax_tab[row][col]
 
            # smaller tick font
            ax.tick_params(labelsize = 9)

            # show 0% strain with vertical dashed line
            ax.axvline(x = 0, color = 'gray', ls = '--')

            # separate upper (trefoil) and lower (pristine) plots
            if row % 2 == 0:

                # merge with lower half by removing bottom spines
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(top = True, bottom = False, **tickKwargs)

                # label Delta E on left axes
                if col == 0:
                    ax.set_ylabel('$\Delta$E (eV)', fontsize = 12)
                    ax.yaxis.set_label_coords(-0.12, -0.05, transform = ax.transAxes)

                # paths to all three types (u, x, y) of strain data
                path_list = [('%s/%s/strain/trefoil/%s' %(top, tmd, label), label_dict[label])\
                             for label in label_dict]

            # lower (pristine) plot
            else:

                # merge with lower half by removing bottom spines
                ax.spines['top'].set_visible(False)
                ax.tick_params(top = False, bottom = True, **tickKwargs)

                # label strain % on bottom axes
                if row == 3:
                    ax.set_xlabel('strain (%)', fontsize = 12)
                    ax.set_xlim(-1.2, 1.2)
            
                # path to biaxial strain data
                path_list = [('%s/%s/strain/unit/u' %(top, tmd), 'biaxial')]

                # label lower axes with TMD species
                title = '%s$_2$' %tmd[:-1]
                ax.text(0.98, 0.02, title, transform = ax.transAxes,
                        va = 'bottom', ha = 'right', fontsize = 14)

                # break in y axis
                transform = ax.transAxes
                ax.plot(x1, y1, transform = transform, **splitKwargs)
                ax.plot(x2, y1, transform = transform, **splitKwargs)
                ax.plot(x1, y2, transform = transform, **splitKwargs)
                ax.plot(x2, y2, transform = transform, **splitKwargs)

                ax.fill_between(x1, y1, y2, transform = transform,
                                color = 'white', clip_on = False, zorder = 3)
                ax.fill_between(x2, y1, y2, transform = transform,
                                color = 'white', clip_on = False, zorder = 3)

            # collect energy / strain data
            for path, label in path_list:
                e_list = []
                s_list = []
                for strain in listdir(path):
                    try:
                        energy = getEnergy('%s/%s/OUTCAR' %(path, strain))
                        strain = float(strain) * 100 - 100
                        e_list.append(energy)
                        s_list.append(strain)
                    except NotADirectoryError:
                        pass

                # trefoil strain: solid lines and assorted colors
                if row % 2 == 0:
                    color = next(ax._get_lines.prop_cycler)['color']
                    ls = '-'

                # pristine strain: dashed lines and tab:blue
                else:
                    edim = 0.0
                    e_list = array(e_list) * 81
                    color = 'tab:green'
                    ls = '--'

                # formation energy per chalcogen vacancy
                print('%s\t%s\ttre: %.6g\tdim: %.6g pris: %.6g'\
                      %(tmd, label[:3], e_list[1], edim, epris))
                e_list = list((array(e_list) + 3*edim - epris) / 6)
                y_list += e_list

                # guess parameters for parabolic fit
                emin, emax = min(e_list), max(e_list)
                guess = (emax, 0, emin)

                # fit curve
                p1, p2, p3 = curve_fit(parabola, s_list, e_list, guess)[0]
                s0 = -p2/(2*p1)
                lower, upper = min(s_list) * 1.1, max(s_list) * 1.1
                S = linspace(lower, upper, 100)
                E = parabola(S, p1, p2, p3)

                # label 0% strain
                ax.axvline(x = 0, color = 'gray', ls = '--')

                # plot data and fitted parabolas
                ax.plot(S, E, ls, label = label, color = color) # best fit
                ax.plot(s_list, e_list, 'o', color = color)     # data points

                # formation energy arrows
                if row % 2 == 0:

                    # top of formation energy arrow
                    if label == 'biaxial':
                        etop = min(E)
                        index = list(E).index(etop)
                        strain = list(S)[index]
                        top_tab[row].append((strain, etop, index))

                else:

                    # bottom of formation energy arrow
                    index = top_tab[row - 1][col][2]
                    ebot = list(E)[index]
                    bot_tab[row - 1].append(ebot)

            # hide tick labels on right column
            if col == 1:
                ax.tick_params(labelleft = 'off')

            # remaining plotting parameters
            if row == 0 and col == 0:
                ax.legend(loc = 2) 

            if row == 0 and col == 1:
                height = max(y_list) - min(y_list) + .14

        # each row shares y-axis
        ymin = min(y_list) - .07
        for col in range(2):
            ax_tab[row][col].set_ylim(ymin, ymin + height)

    # arrow indicating formation energy at trefoil energy minimum
    for row in range(0, 3, 2): # iterate over top halves of plots
        for col in range(2):
            trax = ax_tab[row][col]
            prax = ax_tab[row + 1][col]
            trymin = trax.get_ylim()[0]
            prymax = prax.get_ylim()[1]
            strain, etop, index = top_tab[row][col]
            ebot = bot_tab[row][col]
            height = etop - ebot - (trymin - prymax)
 
            arrowprops = dict(facecolor = 'black', shrink = 0,
                              width = 1.5, headwidth = 8.0, headlength = 10)

            trax.annotate('', xy = (strain, etop), xycoords = 'data',
                          xytext = (strain, trymin), textcoords = 'data',
                          arrowprops = arrowprops, clip_on = False)
                      
            prax.annotate('', xy = (strain, ebot), xycoords = 'data',
                          xytext = (strain, prymax), textcoords = 'data',
                          arrowprops = arrowprops, clip_on = False)

            prax.text(strain + .04 , ebot + height/2, '%.3g eV' %(etop - ebot),
                      fontsize = 11, ha = 'left', va = 'center')

    if save:
        plt.savefig(outfile, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

#-------------------------------- Strain Fig Class -----------------------------
def bM(v, e0, b0, bp, v0):
    """ returns energy predicted by Birch-Murnaghan EOS (PRB 70, 224107) """
    eta = (v0/v)**(2/3)
    E = e0 + 9*b0*v0/16 * (eta - 1)**2 * (6 - 4*eta + bp*(eta - 1))
    return E

def parabola(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

class StrainFig(object):
    """
    Class designed to quickly plot strain energetics
    """
    def __init__(self, data_tab = None, nrows = 2, ncols = 2):
        """
        initializes StrainFig instance with data_tab
        """
        self.nrows = nrows
        self.ncols = ncols

        # holders for data sets (xdata, ydata, label). xdata, ydata are arrays
        if data_tab == None:
            self.data_tab = [[[] for n in range(ncols)] for m in range(nrows)]
        else:
            self.data_tab = deepcopy(data_tab)

    def __add__(self, other):
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n][1]

                    if type(other) == StrainFig:
                        otherE_ar = other.data_tab[row][col][n][1]
                    elif type(other) == list:
                        otherE_ar = other[row][col]
                    elif type(other) == float or type(other) == int:
                        otherE_ar = other
                  
                    selfE_ar += otherE_ar

    def __sub__(self, other):
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n][1]

                    if type(other) == StrainFig:
                        otherE_ar = other.data_tab[row][col][n][1]
                    elif type(other) == list:
                        otherE_ar = other[row][col]
                    elif type(other) == float or type(other) == int:
                        otherE_ar = other
                  
                    selfE_ar -= otherE_ar

    def __mul__(self, other):
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n][1]

                    if type(other) == list:
                        otherE_ar = other[row][col]
                    elif type(other) == float or type(other) == int:
                        otherE_ar = other

                    selfE_ar *= other

    def __truediv__(self, other):
        for row in range(self.nrows):
            for col in range(self.ncols):
                for n in range(len(self.data_tab[row][col])):
                    selfE_ar = self.data_tab[row][col][n][1]
                    selfE_ar /= other

    def addData(self, data = None, label = None, row = 0, col = 0, top = '.'):
        """
        adds data sets to data_tab
        data: (s_list, e_list, label) (tuple)
            if None, data is taken from OUTCARs in top directory
        """
        # collect energies and strain percenteges
        if data == None:
            e_list = []
            s_list = []
            for strain in listdir(top):
                try:
                    energy = getEnergy('%s/%s/OUTCAR' %(top, strain))
                    strain = float(strain) * 100 - 100
                    e_list.append(energy)
                    s_list.append(strain)
                except NotADirectoryError:
                    pass

        # use input data tuple
        else:
            s_list, e_list, label = data

        # enesure that data structures are arrays
        self.data_tab[row][col].append((array(s_list), array(e_list), label))

    def plot(self, figsize = (8, 8), fit = 'p', zero = True):
        """
        plots data and curve fits
        """
        self.plotted = True
        self.fig, self.ax_tab = plt.subplots(figsize = figsize, nrows = self.nrows,
                                                                ncols = self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):

                # prepare axes
                ax = self.ax_tab[row][col]

                for s_list, e_list, label in self.data_tab[row][col]:

                    # UNDER CONSTRUCTION: BM fit
                    if 'b' in fit or 'B' in fit:
                        print('fitting Birch-Murnaghan fit')
                
                        # find curve_fit parameters
                        e0, b0, bp, v0 = curve_fit(bM, v_list, e_list, p0 = guess)[0]
                        print("fitted parameters:\n\
                               E0 = %.4g, B0 = %.4g, B' = %.4g, V0 = %.4g"
                                %(e0, b0, bp, v0))
            
                    # parabolic fit
                    elif 'p' in fit or 'P' in fit:
                        print('fitting parabola')
            
                        # guess parameters
                        if zero:
                            e_list = array(e_list) - e_list[1]
                        guess = (max(e_list), 0, min(e_list))

                        # fit curve
                        p1, p2, p3 = curve_fit(parabola, s_list, e_list, guess)[0]
                        s0 = -p2/(2*p1)
                        lower, upper = min(s_list) * 1.1, max(s_list) * 1.1
                        s = linspace(lower, upper, 100)
                        E = parabola(s, p1, p2, p3)

                        # label 0% strain
                        ax.axvline(x = 0, color = 'gray', ls = '--')

                    # plotting
                    if 'pris' in label:
                        color = 'tab:blue'
                        ls = '--'
                    else:
                        color = next(ax._get_lines.prop_cycler)['color']
                        ls = '-'

                    ax.plot(s, E, label = label, color = color, ls = ls) # best fit
                    ax.plot(s_list, e_list, 'o', color = color)          # data points

                # remaining plot settings
                if row == self.nrows - 1:
                    ax.set_xlabel('strain (%)', fontsize = 12)
                if col == 0:
                    if zero:
                        ax.set_ylabel('$\Delta$E (eV)', fontsize = 12)
                    else:
                        ax.set_ylabel('formation energy (eV)', fontsize = 12)

                ax.tick_params(labelsize = 10)
                ax.locator_params(axis = 'y', nbins = 8)
                ax.locator_params(axis = 'x', nbins = 8)

                ax.set_xlim(lower, upper)
                ax.grid()
                ax.legend()
    
    def setTitle(self, title = 'strain energetics', row = 0, col = 0, loc = 1):
        """
        assigns titles to axes.  Need to have already plotted
        """
        if self.plotted:
            ax = self.ax_tab[row][col]
    
            if loc == 1:
                x, y = .96, .98
                ha, va = 'right', 'top'
            elif loc == 4:
                x, y, = .96, .02
                ha, va = 'right', 'bottom'
    
            ax.text(x, y, title, transform = ax.transAxes,
                    va = va, ha = ha, fontsize = 14)

        else:
            print('Nothing has been plotted. Run instance.plot() first.')

    def combinePlots(self, other):
        """
        adds data and lines from other StrainFig instance
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                selfax = self.ax_tab[row][col]
                otherax = other.ax_tab[row][col]

                selfax.lines += otherax.lines
        
    def copy(self):
        return StrainFig(self.data_tab, self.nrows, self.ncols)

    def save(self, outfile = 'testStrain.pdf'):
        self.fig.tight_layout()
        self.fig.savefig(outfile)

    def show(self):
        self.plotted = False
        self.fig.tight_layout()
        self.fig.show()
