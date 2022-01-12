# plot local potential and/or charge density along certain direciton
# Anthony Yoshimura
# 11/21/16

# requires LOCPOT or CHGCAR

from numpy import array, reshape, linspace, polyfit, where
import matplotlib.pyplot as plt
import os

# include date on outfile file names
from datetime import date
from shutil import copyfile

# open with wsl
from subprocess import call


def plotS(
        outfile='SLPZ.pdf',
        root = '/home/yoshia/research/ornl/atoms/lda/S',
        view = False,
        ):
    plotCompLPZ(outfile=outfile, root=root, view=view)


def plotB(
        outfile = 'BLPZ.pdf',
        root = '/home/yoshia/research/ornl/atoms/lda/B',
        view = False,
        ):
    plotCompLPZ(outfile=outfile, root=root, view=view)


def plotCompLPZ(
        outfile='BLPZ.pdf',
        root = '/home/yoshia/research/ornl/atoms/lda/B',

        colormap = 'plasma',
        offset = 5,
        multiplier = 10,

        xbounds = None,
        ybounds = None,
        view = True,
        
        subtractFermi = False,
        grid = True,
        transparent = False,
        legend = True,
        ):
    """
    plots avg loc pot vs z for isolated boron
    """
    Plot = LPZPlot()

    d_list = next(os.walk(root))[1]
    cmap = plt.get_cmap(colormap, 10*len(d_list))
    for n, d in enumerate(d_list):
        dLOCPOT='{}/LOCPOT'.format(d)
        dOUTCAR='{}/OUTCAR'.format(d)
        print('reading {}'.format(dLOCPOT))
        Plot.addLPZ(LOCPOT=dLOCPOT, OUTCAR=dOUTCAR, root=root, label=d,
                color=cmap(multiplier*n + offset))

    Plot.decorate(xbounds=xbounds, ybounds=ybounds, legend=legend, grid=grid,
            subtractFermi=subtractFermi)
    Plot.save(outfile, view=view, transparent=transparent)
    Plot.close()


class LPZPlot(object):
    """
    class to add avg local potential vs. Z curves to single figure
    """
    def __init__(self, fig=None, ax=None, figsize=(6,5)):
        """
        intializes figure and axes
        """
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def addLPZ(self,
            LOCPOT = 'LOCPOT',
            OUTCAR = 'OUTCAR',
            root = '/home/yoshia/ornl/atoms/lda/S',
            label = 1,
            color = 'tab:green',
            linestyle = '-',
            subtractFermi = True,
            annotate = False,
            ):
        """
        plots avg loc pot as a function of z from to
        """
        zpot_ar, z_ar, vacpot, vacpot_z = getLPZ('{}/{}'.format(root, LOCPOT))

        if subtractFermi:
            efermi = getEFermi('{}/{}'.format(root, OUTCAR))
            zpot_ar -= efermi
            vacpot -= efermi

        self.ax.plot(z_ar, zpot_ar, lw=2, label=label, color=color,
                ls=linestyle)

        if annotate:
            self.ax.annotate('%.3g eV' %vacpot,
                    fontsize=14, ha='center',
                    xy=(vacpot_z, vacpot), xycoords='data',
                    xytext=(0, -40), textcoords='offset points',
                    arrowprops=dict(facecolor='k', shrink=0.05))


    def decorate(self,
            xbounds = None,
            ybounds = None,
            xlabel = r'$z$ $(\AA)$',
            ylabel = 'energy (eV)',
            title = None,
            forslides = False,
            xticks = [0, 50, 100],
            yticks = [0, 1],
            legend = True,
            grid = True,
            subtractFermi = True,
            ):
        """
        adds plot attributes
        """
        if xbounds == None:
            self.ax.set_xmargin(0)
        else:
            self.ax.set_xlim(xbounds)

        if ybounds != None:
            self.ax.set_ylim(ybounds)

        self.ax.set_title(title, fontsize = 20)
        self.ax.set_xlabel(r'$z$ $(\AA)$', fontsize = 18)
        if subtractFermi:
            ylabel = r'$V - E_F$ (eV)'
        else:
            self.ax.set_ylabel(ylabel, fontsize = 18)

        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14, loc=1)
        plt.tight_layout()

    def save(self,
            outfile = 'BLPZ.pdf',
            dest = '.',
            view = True,
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        plots probability as a function of beam energy from to
        dest: directory to which plot is saved (str)
        """
        if dest == '.':
            outpath = outfile
        else:
            outpath = '{}/{}'.format(dest, outfile)
        plt.savefig(outpath, dpi=dpi, transparent=transparent)
    
        # make copy with date in name
        if writedate:
            today = date.today()
            year = today.year - 2000
            month = today.month
            day = today.day
        
            name, ext = outpath.split('.')
            outpathcopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month, day, ext)
        
            copyfile(outpath, outpathcopy)

        if view:
            call('wsl-open {}'.format(outpath), shell=True)

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return LPZPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()


def getLPZ(LOCPOT='LOCPOT'):
    """
    returns average local potential with respect to z for unit cell
    LOCPOT: LOCPOT file (str)
    """
    with open(LOCPOT) as f:

        for n in range(4):
            f.readline()
    
        height = float(f.readline().split()[-1])
        
        f.readline()
    
        pop = sum([int(val) for val in f.readline().split()])
        
        for n in range(pop + 2):
            f.readline()
    
        ngx, ngy, ngz = [int(val) for val in f.readline().split()]
        ng = ngx * ngy * ngz
    
        pot_ar = array([float(val) for val in f.read().split()[:ng]])
        pot_a3 = reshape(pot_ar, (ngz, ngy, ngx))
        zpot_ar = pot_a3.mean(axis=(1,2))

    z_ar = linspace(0, height, ngz)
    vacpot = max(zpot_ar)
    vacpot_index = where(zpot_ar==vacpot)
    vacpot_z = float(z_ar[vacpot_index])
    
    print('vacuum potential = {:.6g} at z = {:.6g}'.format(vacpot, vacpot_z))

    return zpot_ar, z_ar, vacpot, vacpot_z


def getEFermi(OUTCAR='OUTCAR'):
    """
    returns fermi energy
    """
    with open(OUTCAR) as inputFile_OUTCAR:
        for line in inputFile_OUTCAR:
            if 'E-fermi' in line:
                efermi = float(line.split()[2])

    return efermi

