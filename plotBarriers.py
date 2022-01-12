# Anthony Yoshimura
# 08/07/17

#from scipy.interpolate import interp1d
from numpy import pi, log, arange, e, cos, sin, sqrt, linspace, zeros, floor, ceil
from numpy import array, transpose, dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from plotEnCon import getPaths, getEnergy, getEnergies
import os
import POSCAR as p

def plot(
        # input files
        top = '.',
        mustContain = None,

        # inputs
        label_list = [],

        # plot attributes
        figsize = (10, 6),
        ratio = (3, 1),
        title = None,
        color_list = 'auto',
        show_formation = False,
        show_bars = True,
        ebounds = None,

        # saving
        save = False,
        dest = '.',
        outfile = 'barriers.pdf',
        resolution = 200,
        spline = False,

        # outputes
        get_raw_data = False,
        printInfo = False,
        ):
    """
    plots energy barriers from NEB. Run from lowest common root
    energy_tab: table of energies
    top: top-level directory (str)
    label_list: labels for energy profile legend and bar chart
    get_raw_data: if True, writes energy data to raw_data.txt
    """
    # collect paths to neb runs
    path_list = getPaths(mustContain = mustContain, top = top)
    if printInfo:
        print('path_list: %s' %path_list)

    # get list of systems
    spec_list = []
    for path in path_list:
        dir_list = path.split('/')

        # recognize energies belonging to different groups (or specs)
        if len(dir_list) > 3:
            spec = dir_list[1] 
            # traj = dir_list[3]
            traj = dir_list[-3]
        else:
            spec = dir_list[0] 
            traj = spec

        if (spec, traj) not in spec_list:
            spec_list.append((spec, traj))

    if printInfo:
        print('spec_list: %s' %spec_list)

    # extract energies from paths
    energy_tab = []
    for spec, traj in spec_list:

        energy_list = []
        for path in path_list:

            if spec + '/' in path and traj + '/' in path:
                energy = getEnergy(path)
                energy_list.append(energy)

        energy_ar = array(energy_list) - energy_list[0]
        energy_tab.append(energy_ar)

    # energy profile domains
    print('path_list: %s' %path_list)
    print('energy_tab: %s' %energy_tab)
    images = [n for n in range(len(energy_tab[0]))]
    domain = linspace(0, len(energy_tab[0]) - 1, resolution)

    # create figure
    fig = plt.figure(figsize = figsize)

    # set axes for profiles and barriers if there are more than one species
    if len(spec_list) > 1:
        gs = gridspec.GridSpec(1, ratio[0] + ratio[1])
        ax1 = fig.add_subplot(gs[0, :ratio[0]])   # profiles take up 2/3
        ax2 = fig.add_subplot(gs[0, -ratio[1]], sharey = ax1)
        for label in ax2.get_yticklabels():
            label.set_visible(False)       # don't renumber energy axis on bar plot
        fig.subplots_adjust(wspace = 0.05) # less space between subplots

    # set axes for profiles with barriers shown as arrows
    else:
        ax1 = fig.add_subplot(111)

    # use default colors from matplotlib
    if color_list == 'auto':
        color_list = [next(ax1._get_lines.prop_cycler)['color'] for n in range(len(energy_tab))]

    # find kinetic energy barriers
    kbarriers = []
    kbounds = []
    if printInfo:
        print('length of energy table = %s' %len(energy_tab))

    for energy_list in energy_tab:
        if printInfo:
            print('length of energy list = %s' %len(energy_list))

        # find largest barrier
        maxkdiff = 0
        for n in range(1, len(energy_list)):
            for m in range(n - 1):
                kdiff = energy_list[n] - energy_list[m]
                if kdiff > maxkdiff:
                    maxkdiff = kdiff
                    emax = energy_list[n]
                    emin = energy_list[m]
        kbarriers.append(maxkdiff)
        kbounds.append((emin, emax))

    # find thermodynamic barrier
    tbarriers = [energy_list[-1] for energy_list in energy_tab]

    # plot kinetic barriers as bars
    if len(spec_list) > 1 and show_bars:
        num_specs = len(energy_tab)
        width = .3

        for n in range(num_specs):
            height = kbarriers[n]
            emin, emax = kbounds[n]
            color = color_list[n]
#            ax2.bar(left = arange(num_specs) / 3, height = height, width = width,
#                    bottom = emin, color = color, edgecolor = color, zorder = 2)
#            ax2.bar(left = n / 3, height = height, width = width,
#                    bottom = emin, color = color, edgecolor = color, zorder = 2)
            ax2.bar(n / 3, height = height, width = width,
                    bottom = emin, color = color, edgecolor = color, zorder = 2)
        if show_formation:
            ax2.bar(arange(num_specs) / 3, tbarriers, width, color = "white",\
                    edgecolor = color_list, hatch = "//", zorder = 2)

    # or show barriers with arrows
    else:
        kbarrier = kbarriers[0]
        max_index = list(energy_list).index(max(energy_list))
        ax1.annotate('', xy = (max_index, kbarrier), xytext = (max_index, 0),
                    xycoords = 'data', textcoords = 'data',
                    arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                    shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
        ax1.annotate('',
                    xy = (max_index, 0),
                    xytext = (max_index, kbarrier),
                    xycoords = 'data', textcoords = 'data',
                    arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                    shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 

        xpos = max_index + .03
        ypos = kbarrier / 2
        halign = 'left'
        valign = 'center'

        ax1.text(xpos, ypos, '%.3g eV' %(kbarrier), fontsize = 16,
                 verticalalignment = valign, horizontalalignment = halign)

        if show_formation:
            tbarrier = tbarriers[0]
            last_index = len(energy_list) - 1
            ax1.annotate('', xy = (last_index, tbarrier), xytext = (last_index, 0),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
            ax1.annotate('',
                        xy = (last_index, 0),
                        xytext = (last_index, tbarrier),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
    
            xpos = last_index - .03
            ypos = tbarrier / 2
            halign = 'right'
            valign = 'center'
    
            ax1.text(xpos, ypos, '%.3g eV' %(tbarrier), fontsize = 16,
                     verticalalignment = valign, horizontalalignment = halign)

    # labels for profiles and barriers
    if label_list == 'auto':
        pass
#        if spec_list
    if len(label_list) == 0:
        label_list = ['%s$_2$' %spec[0][:-1] for spec in spec_list]
   
    # plot energy profiles
#    for spec in range(len(energy_tab)):
    for n, color, label, energy_list, kbarrier, tbarrier in\
        zip(range(len(energy_tab)), color_list, label_list, energy_tab, kbarriers, tbarriers): 
        print('system = %s' %label)

        # energy profiles with cubic interpolation
        ax1.plot(images, energy_list, 'o', color = color, mec = color)
        if spline:
            cs = interp1d(images, energy_list, kind = 'cubic')
            ax1.plot(domain, cs(domain), linewidth = 2, color = color, label = label)
        else:
            print('images : %s,\nenergies : %s' %(images, energy_list))
            ax1.plot(images, energy_list, color = color, mec = color, linewidth = 2, label = label)

        # show values of kbarriers
        if len(spec_list) > 1:
            ax2.text(n / 3, kbarrier,
                     '%.3g' %(kbarrier), fontsize = 12, ha = 'center', va = 'bottom')

        # show values of tbarriers
        offset = -.25
        color = 'black'
        if tbarrier > 0:
            offset = .05
            color = 'white'
        if show_formation and len(spec_list) > 1:
            print('going to ax2')
#            ax2.text(spec / 3, tbarriers[spec] + offset, str(round(tbarriers[spec], 2)),
            ax2.text(n / 3, tbarrier + offset, str(round(tbarrier, 2)),
                     fontsize = 12, ha = 'center', color = color)

    # energy barrier subplot attributes
    if len(spec_list) > 1:
        ax2.axhline(0, color = 'black')
        ax2.grid()
        ax2.set_xticks(arange(num_specs) / 3)
        ax2.set_xticklabels(label_list, rotation = 'vertical', fontsize = 12)
    
    # energy profile subplot attributes
    if type(title) == str:
        ax1.set_title(title, fontsize = 18)
    ax1.set_xlabel('reaction coordinate', fontsize = 18)
    ax1.set_ylabel('energy (eV)', fontsize = 18)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax1.set_xlim(0, len(energy_list) - 1)
    if ebounds != None:
        ax1.set_ylim(ebounds)

    ax1.axhline(0, color = 'black')
    ax1.grid()
    if len(spec_list) > 1:
        ax1.legend(loc = 2, fontsize = 14)

    # save figures
    plt.tight_layout()
    if save:
        plt.savefig('%s/%s' %(dest, outfile))
    plt.show()

    # write tabulated energy data to file
    if get_raw_data:
        with open('raw_data.txt', 'w') as f:
            for spec in range(len(spec_list)):
                f.write('%s\t' %spec_list[spec][0])
                for energy in energy_tab[spec]:
                    f.write('%s\t' %energy)
                f.write('\n')


def getDistances(end, atom, totalDist = None, percentage = True):
    """
    returns list of distances of a particular atom from its final position
    end: instance of POSCAR class, to which all POSCARs in current directory are compared (POSCAR)
    atom: atom as indexed in VESTA whose position will be compared (pos int)
    """
    atom -= 1
    path_list = ['%s/POSCAR' %f for f in os.listdir() if f[:2].isdigit()]

    if totalDist == None:
        start = p.POSCAR(path_list[0])
        totalDist = norm(array(start.coords[atom]) - array(end.coords[atom]))

    dist_list = []
    for path in path_list:
        pos = p.POSCAR(path)
        remainingDist = norm(array(pos.coords[atom]) - array(end.coords[atom]))
        dist = totalDist - remainingDist
        if percentage:
            dist /= totalDist
        dist_list.append(dist)

    return dist_list


def plotFromTable(
        energy_tab,
        label_list,
        source = 'table',
        reference = 'start',
        barrierDisplay = 'bars',
        figsize = (10, 6),
        ratio = (3, 1),
        title = None,
        show = True,
        save = False,
        outfile = 'barriers.pdf',
        color_list = ['blue', 'green', 'red', 'orange'],
        labelHeight_list = [.5, .7, .5, .4],
        resolution = 200,
        show_formation = False,
        ebounds = None,
        mustContain = None,
        top = '.',
        spline = False,
        get_raw_data = False,
        ):
    """
    plots energy barriers from NEB. Run from lowest common root
    energy_tab: table of energies
    top: top-level directory (str)
    label_list: labels for energy profile legend and bar chart
    get_raw_data: if True, writes energy data to raw_data.txt
    """
    if source == 'OUTCAR':
        pass

    elif source == 'table':
        new_energy_tab = []
        for energy_list in energy_tab:
            energy_ar = array(energy_list[:])
            if reference == 'end':
                energy_ar -= energy_list[-1]
            else:
                energy_ar -= energy_list[0]
            energy_list = energy_ar.tolist()
            new_energy_tab.append(energy_list)
            
        energy_tab = new_energy_tab
    
    # domains
    image_tab = [[n * 100 / (len(energy_tab[e]) - 1) for n in range(len(energy_tab[e]))] for e in range(len(energy_tab))]
    domain_tab = [[linspace(0, len(energy_tab[0]) - 1, resolution)] for e in range(len(energy_tab))]

    # create figure
    fig = plt.figure(figsize = figsize)
    if barrierDisplay == 'bars':
        gs = gridspec.GridSpec(1, ratio[0] + ratio[1])
        ax1 = fig.add_subplot(gs[0, :ratio[0]])   # profiles take up 2/3
        ax2 = fig.add_subplot(gs[0, -ratio[1]], sharey = ax1)
        for label in ax2.get_yticklabels():
            label.set_visible(False)       # don't renumber energy axis on bar plot
        fig.subplots_adjust(wspace = 0.05) # less space between subplots
    else:
        ax1 = fig.add_subplot(111)
    if ebounds != None:
        ax1.set_ylim(ebounds)

    # find kinetic energy barriers
    kbarriers = []
    kbounds = []
    indexPairs = []
    for energy_list in energy_tab:
        maxkdiff = 0
        for n in range(1, len(energy_list)):
            for m in range(n - 1):
                kdiff = energy_list[n] - energy_list[m]
                if kdiff > maxkdiff:
                    maxkdiff = kdiff
                    emax = energy_list[n]
                    emin = energy_list[m]
                    pair = (m, n)
        kbarriers.append(maxkdiff)
        kbounds.append((emin, emax))
        indexPairs.append(pair)

    tbarriers = [energy_list[-1] for energy_list in energy_tab]

    # plot energy bars
    if barrierDisplay == 'bars':
        num_labels = len(energy_tab)
        width = .3
        for n in range(num_labels):
            height = kbarriers[n]
            emin, emax = kbounds[n]
            color = color_list[n]
            ax2.bar(left = n / 3, height = height, width = width,
                    bottom = emin, color = color, edgecolor = color, zorder = 2)
        if show_formation:
            ax2.bar(arange(num_labels) / 3, tbarriers, width, color = "white",
                    edgecolor = color_list, hatch = "//", zorder = -2)

    # or show barriers with arrows
    elif barrierDisplay == 'arrows':
        num_labels = len(energy_tab)
        for n in range(len(energy_tab)):
            kbarrier = kbarriers[n]
            emin, emax = kbounds[n]
            minIndex, maxIndex = indexPairs[n]
            color = color_list[n]
#            color = 'k'
            color1 = 'k'
#            color1 = color
            labelHeight = labelHeight_list[n]

            minPoint, maxPoint = image_tab[n][minIndex], image_tab[n][maxIndex]

            # up arrow
            ax1.annotate('', xy = (maxPoint, emax), xytext = (maxPoint, emin),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = color, edgecolor = color,
                        # arrowstyle = '<->',
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 

            # down arrow
            ax1.annotate('',
                        xy = (maxPoint, emin), xytext = (maxPoint, emax),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = color, edgecolor = color,
                        # arrowstyle = '<->',
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 

            # horizontal line connecting down arrow tip to energy minimum
            ax1.plot([minPoint, maxPoint], [emin, emin], ls = '--', color = color)
    
            halign = 'left'
            valign = 'center'
            xpos = maxPoint + .3
            ypos = (emax - emin) * labelHeight + emin
    
            ax1.text(xpos, ypos, '%.2f' %(kbarrier), fontsize = 15, color = color1,
                     verticalalignment = valign, horizontalalignment = halign)
    
            if show_formation:
                tbarrier = tbarriers[0]
                last_index = len(energy_list) - 1
                ax1.annotate('', xy = (last_index, tbarrier), xytext = (last_index, 0),
                            xycoords = 'data', textcoords = 'data',
                            arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                            shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
                ax1.annotate('',
                            xy = (last_index, 0),
                            xytext = (last_index, tbarrier),
                            xycoords = 'data', textcoords = 'data',
                            arrowprops = dict(facecolor = 'k', edgecolor = 'k',# arrowstyle = '<->',
                            shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
        
                xpos = last_index + .05
                ypos = tbarrier / 2
                halign = 'right'
                valign = 'center'
        
                ax1.text(xpos, ypos, '%.3g eV' %(tbarrier), fontsize = 16,
                         verticalalignment = valign, horizontalalignment = halign)

    # plot energy profiles
    for n in range(len(energy_tab)):
        color = color_list[n]
        energy_list = energy_tab[n]
        image_list  = image_tab[n]
        domailabel_list = domain_tab[n]

        # energy profiles with cubic interpolation
        lab = label_list[n] # I stupidly used 'n' as the number index
        ax1.plot(image_list, energy_list, 'o', color = color, mec = color)
        if spline:
            cs = interp1d(image_list, energy_list, kind = 'cubic')
            ax1.plot(domain_list, cs(domain_list), linewidth = 2, color = color, label = lab)
        else:
            ax1.plot(image_list, energy_list, color = color, mec = color, linewidth = 2, label = lab)

        # show values of kbarriers
        if barrierDisplay == 'bars':
            kbarrier = kbarriers[n]
            emin, emax = kbounds[n]
            va = 'bottom'
            offset = 0
            color = 'black'
            if emax < 0:
                va = 'top'
                offset = -.01
                color = 'white'
            ax2.text(n / 3, emax + offset,
                     '%.2f' %(kbarrier), fontsize = 12, ha = 'center', va = va, color = color)

            # show values of tbarriers
            offset = -.1
            color = 'black'
            if tbarriers[n] > 0:
                offset = .05
                color = 'white'
            if show_formation and len(label_list) > 1:
                print('going to ax2')
    #            ax2.text(n / 3, tbarriers[n] + offset, str(round(tbarriers[n], 2)),
                ax2.text(n / 3, tbarriers[n] + offset, str(round(tbarriers[n], 2)),
                         fontsize = 12, ha = 'center', color = color)

            # energy barrier subplot attributes
            ax2.axhline(0, color = 'black')
            ax2.grid()
            ax2.set_xticks(arange(num_labels) / 3)
            ax2.set_xticklabels(label_list, rotation = 'vertical', fontsize = 12)
    
    # energy profile subplot attributes
    if type(title) == str:
        ax1.set_title(title, fontsize = 18)
    ax1.set_xlabel('percentage of path traversed', fontsize = 18)
    ax1.set_ylabel('energy (eV)', fontsize = 18)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax1.set_xlim(0, 100)
    if reference == 'end':
        ax1.set_ylim(0, 2)
    ax1.axhline(0, color = 'black')

    ax1.grid()
    if len(label_list) > 1:
        ax1.legend(loc = 'best', fontsize = 14)

    plt.tight_layout()
    if save:
        plt.savefig(outfile)
    if show:
        plt.show()

    if get_raw_data:
        with open('raw_data.txt', 'w') as f:
            for label in range(len(label_list)):
                f.write('%s\t' %label_list[label][0])
                for energy in energy_tab[label]:
                    f.write('%s\t' %energy)
                f.write('\n')

#------------------------------- broken path ----------------------------------
def getInitSeg_list(top = '.', mustContain = 'pa_e5_f'):
    """
    returns list of (e.g. vacancy) sites connected by migration paths
    top: top-level directory containing NEB runs labelled as ___to___ (str)
    mustContain: name of directory that must be contained in path (str)
    """
    # collect paths to neb runs
    path_list = getPaths(mustContain = mustContain, top = top)
    depth = len(top.split('/')) - 1

    # get list of distinct directoties
    dir_list = []
    seg_list = []
    for path in path_list:
        directory = path.split('/')[1 + depth]
        if directory not in dir_list and 'old' not in directory:
            dir_list.append(directory)
            start, end = directory.split('to')
            seg_list.append((start, end)) # use tuples becuase they are hashable

    return seg_list


def joinSegments(seg_list = 'auto', top = '.', mustContain = 'pa_e5_f'):
    """
    joins segments with matching starts and ends
    seg_list: list of segments (list of tuples of strings)
    """
    if seg_list == 'auto':
        seg_list = getInitSeg_list(top = top, mustContain = mustContain)

    if len(seg_list) < 2:
        print('finished joining!')
        return seg_list

    jointSeg_list = [] # list of joined segments
    usedSeg_list = [] # list of segments that have already been joined
    for seg1 in seg_list:

        # ignore segment that has already been joined
        if seg1 in usedSeg_list:
            pass
        else:

            # check if beginning of another segment matches end of seg1
            seg2_list = seg_list[:]
            seg2_list.remove(seg1)

            for seg2 in seg2_list:

                # ignore segment that has already been joined
                if seg2 in usedSeg_list:
                    pass
                else:

                    # join segments whose ends match
                    jointSeg = seg1
                    if seg1[-1] == seg2[0]:
                        jointSeg = seg1[:-1] + seg2
                        usedSeg_list += [seg1, seg2] # don't check segs again
                        break
    
            jointSeg_list.append(jointSeg)

    # remove segments that were joined
    for usedSeg in usedSeg_list:
        try:
            jointSeg_list.remove(usedSeg)
        except ValueError:
            pass

    # remove duplicate joint segments
    jointSeg_list = list(set(jointSeg_list))
    jointSeg_list.sort() # preserve order in case set{jointSeg_list} = set{seg_list}

    # show steps
    print('removed %s' %usedSeg_list)
    print('jointSeg_list: %s' %jointSeg_list)

    # check if joined all segments
    if jointSeg_list == seg_list:
        print('finished joining!')
        return jointSeg_list

    # repeat if more segments have been joined
    else:
        return joinSegments(jointSeg_list)


def getEnergiesForSegs(jointSeg_list = 'auto', top = '.', mustContain = 'pa_e5_f'):
    """
    returns list of lists of energies along joint segments
    """
    if jointSeg_list == 'auto':
        initSeg_list = getInitSeg_list(top = top, mustContain = mustContain)
        jointSeg_list = joinSegments(initSeg_list)
        
    # get energies along to joint-segments        
    en_dict = {}
    for jointSeg in jointSeg_list:

        # longEn: along full joint-segment
        longEn_list = []
        numJoints = len(jointSeg) - 1
        for joint in range(numJoints):

            # shortEn: along individual segment
            start = jointSeg[joint]
            end = jointSeg[joint + 1]
            newTop = '%s/%sto%s/%s' %(top, start, end, mustContain)

            print('extracting energies from %s' %newTop)
            if joint == numJoints - 1:
                print('adding last segment')
                path_list = getPaths('OUTCAR', top = newTop)
            else:
                # end of this seg = start of next seg
                path_list = getPaths('OUTCAR', top = newTop)[:-1]

            # get energies from OUTCARs
            shortEn_list = [getEnergy(path) for path in path_list]
            longEn_list += shortEn_list

        # add key : value pairs
        key = r'%s$\rightarrow$%s' %(jointSeg[0], jointSeg[-1])
        en_dict[key] = longEn_list

    return en_dict

def plotSegmentProfiles(energy_dict = 'auto',

        # extraction directories
        top = '.',
        mustContain = 'pa_e5_f',

        # plot attributes
        title = None,
        figsize = (10, 6),
        show_formation = False,
        show_bars = True,
        ratio = (3, 1),
        ebounds = None,
        color_list = 'auto',
        label_list = 'auto',
        spline = False,
        resolution = 200,

        # saving
        save = False,
        outfile = 'barriers.pdf',
        dest = '.',

        # debugging
        printInfo = False,
        get_raw_data = False,
        ):
    """
    Plots separate NEB runs as one contunuous profile
        * NEB runs must be labelled by start and end position ___to___
        * recognizes adjacent segments if end of one path is start of another
        * full paths cannot share intermediate points
        * ignores NEB run if 'old' is in directory name
    top: top-level directory containing NEB runs labelled as ___to___ (str)
    atom: migrating atom, to track percentage of path traversed (pos int)
    dest: directory to which plot is saved
    label_list: labels for energy profile legend and bar chart
    get_raw_data: if True, writes energy data to raw_data.txt
    """
    if energy_dict == 'auto':
        energy_dict = getEnergiesForSegs()

    # create figure
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(111)

    # use default colors from matplotlib
    if color_list == 'auto':
        color_list = [next(ax1._get_lines.prop_cycler)['color'] for n in range(len(energy_dict))]

    # plot energy profiles (with respect to distance)
    for label, color in zip(energy_dict, color_list):
        energy_ar = array(energy_dict[label])
        energy_ar -= energy_ar[-1]

        # energy profiles with cubic interpolation
        ax1.plot(energy_ar, 'o', color = color, mec = color)
        ax1.plot(energy_ar, color = color, mec = color, linewidth = 2, label = label)

    # energy profile subplot attributes
    if type(title) == str:
        ax1.set_title(title, fontsize = 18)
    ax1.set_xlabel('reaction coordinate', fontsize = 18)
    ax1.set_ylabel('energy (ev)', fontsize = 18)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax1.set_xlim(0, len(energy_ar) - 1)
    if ebounds != None:
        ax1.set_ylim(ebounds)

    ax1.axhline(0, color = 'black')
    ax1.legend(loc = 1, fontsize = 14)
    ax1.grid()

    # save figures
    plt.tight_layout()
    if save:
        plt.savefig('%s/%s' %(dest, outfile))
    plt.show()

    # write tabulated energy data to file
    if get_raw_data:
        with open('raw_data.txt', 'w') as f:
            for spec in range(len(spec_list)):
                f.write('%s\t' %spec_list[spec][0])
                for energy in energy_tab[spec]:
                    f.write('%s\t' %energy)
                f.write('\n')

def getTops(returnTop = 'neb', top = '.', system_list = ['BaZrS3', 'MAPbI3']):
    """
    returns list of directories containing neb runs
    returnTop: only include calculations from neb directories (str)
    """
    # collect paths to neb directories
    path_list = []
    for root, dirs, files in os.walk(top):
        for system in system_list:
            if system in root:
                root_list = root.split('/')
                if root_list[-1] == returnTop:
                    path_list.append(root)

    return path_list
    

def getBars(energy_dict = 'auto', top = '.', mustContain = 'pa_e5_f'):
    """
    returns dictionary containing energy barriers
    """
    if energy_dict == 'auto':
        energy_dict = getEnergiesForSegs(top = top, mustContain = mustContain)

    barrier_dict = {}
    for key in energy_dict:
        profile = energy_dict[key]
        barrier = max(profile) - profile[0]
        barrier_dict[key] = barrier

    return barrier_dict

    
def plotBarriers(barrier_dict = 'auto',

        # extraction directories
        top = '/Users/anthonyyoshimura/Desktop/koratkar/',
        system_list = ['BaZrS3', 'MAPbI3'],
        mustContain = 'pa_e5_f',
        returnTop = 'labelledNeb',

        # plot attributes
        title = None,
        figsize = None,
        ebounds = None,
        color_list = 'auto',
        label_list = 'auto',
        resolution = 200,

        # saving
        save = False,
        outfile = 'BarrierBars.pdf',
        dest = '/Users/anthonyyoshimura/Desktop/koratkar/BaZrS3/plots',

        # debugging
        printInfo = False,
        get_raw_data = False,
        ):
    """
    plots barriers as bar plot
    """
    # keep track of number of nebs per system
    boundary_list = []

    # get barriers from neb runs
    if barrier_dict == 'auto':
        barrier_dict = {}
        newTop_list = getTops(returnTop = returnTop, top = top, system_list = system_list)

        for newTop in newTop_list:
            new_dict = getBars(top = newTop, mustContain = mustContain)
            boundary_list.append(sum(boundary_list) + len(new_dict))
            barrier_dict = {**barrier_dict, **new_dict}

    # boundaries labelled by paths
    if label_list == 'auto':
        label_list = [label for label in barrier_dict]

    # create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)

    # color of bars grouped by system
    if color_list == 'auto':
        color_list = [next(ax._get_lines.prop_cycler)['color'] for n in range(10)]
    
    boundaryIndex = 0
    colorIndex = 0
    colors = []
    for n in range(len(barrier_dict)):
        color = color_list[colorIndex]
        colors.append(color_list[colorIndex])
        if n + 1 >= boundary_list[boundaryIndex]:
            boundaryIndex += 1
            colorIndex += 1

    # plot bars
    for n, label in enumerate(barrier_dict):
        color = colors[n]
        barrier = barrier_dict[label]
        ax.bar(n, barrier, color = color)

    # figure properties
    ax.set_ylim(ebounds)
    ax.set_xlabel('migration path', fontsize = 12)
    ax.set_ylabel('energy (eV)', fontsize = 12)
    ax.set_xticks(arange(len(barrier_dict)))

    if max([len(str(label)) for label in label_list]) > 5:
        ax.set_xticklabels(label_list, fontsize = 10, rotation = 'vertical')
    else:
        ax.set_xticklabels(label_list, fontsize = 10)
#
    # save figure
    plt.tight_layout()
    if save:
         plt.savefig('%s/%s' %(dest, outfile))
    plt.show()


#--------------------------------- For Candidacy --------------------------------

def getEnTab(
        top = '/Users/anthonyyoshimura/Desktop/koratkar/ReS2/neb/cineb',
        ref = 'end',
        ):
    """
    returns energy table for ReS2 nebs
    top: top level directory for energy extraction (str)
    ref: reference energy position ('start' or 'end')
    """
    # get energies from cineb runs
    en_dict = getEnergies(top = top, mustContain = 'ph_e5', printPaths = False)

    # names of paths
#    path_tab = [['A1toA1p', 'A1ptoA1pp', 'A1pptoB'],
#                 ['A2toA2p', 'A2ptoA2pp', 'A2pptoB'],
#                 ['A3toA1pp', 'A1pptoB'],
#                 ['A4toA4p', 'A4ptoA1pp', 'A1pptoB']]
#    path_tab = [['A4toA4p', 'A4ptoA1pp', 'A1pptoB']]

    # combine paths based on name
    en_tab = []
    for path_list in path_tab:
        en_list = []
        for path in path_list:
            en_list += en_dict[1][path]
        en_tab.append(en_list)

    # remove overlapping energies
    new_en_tab = []
    for en_list in en_tab:
        en_ar = array(en_list[:])
        if ref == 'end':
            en_ar -= en_list[-1]
        else:
            en_ar -= en_list[0]
        en_list = en_ar.tolist()
        new_en_tab.append(en_list)
        
    return new_en_tab


def plotWhite(
        label_list = ['A1', 'A2', 'A3', 'A4'],
        top = '/Users/anthonyyoshimura/Desktop/koratkar/ReS2/neb/cineb',
        source = 'table',
        reference = 'start',
        barrierDisplay = 'bars',
        figsize = None,
        save = False,
        outfile = 'whiteProfile.pdf',
        color_list = 'auto',
        labelHeight_list = [.5, .7, .5, .4],
        resolution = 200,
        ebounds = None,
        mustContain = None,
        spline = False,
        get_raw_data = False,
        arrows = True,
        barriers = False,
        en_tab = 'auto',
        ):
    """
    plots energy barriers from NEB. Run from lowest common root
    energy_tab: table of energies
    top: top-level directory (str)
    label_list: labels for energy profile legend and bar chart
    get_raw_data: if True, writes energy data to raw_data.txt
    """
    # get energies along each path
#    if en_tab == 'auto':
#        en_tab = getEnTab(top = top)
#        for m in range(len(en_tab)):
#            for n in range(len(en_tab[m]) - 1):
#                en_tab[m][n] += .08
    
    # domain (reaction coordinates)
    image_tab = [[n * 100 / (len(en_tab[e]) - 1)
                 for n in range(len(en_tab[e]))]
                 for e in range(len(en_tab))]

    # create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)

    # use default colors from matplotlib
    if color_list == 'auto':
        color_list = [next(ax._get_lines.prop_cycler)['color'] for n in range(len(en_tab))]

    # find kinetic energy barriers
    kbarriers = []
    kbounds = []
    indexPairs = []
    for en_list in en_tab:
        maxkdiff = 0
        for n in range(1, len(en_list)):
            for m in range(n - 1):
                kdiff = en_list[n] - en_list[m]
                if kdiff > maxkdiff:
                    maxkdiff = kdiff
                    emax = en_list[n]
                    emin = en_list[m]
                    pair = (m, n)
        kbarriers.append(maxkdiff)
        kbounds.append((emin, emax))
        indexPairs.append(pair)

    # show kinetic barriers as arrows
    if arrows:
        num_labels = len(en_tab)
        for n in range(len(en_tab)):
            kbarrier = kbarriers[n]
            emin, emax = kbounds[n]
            minIndex, maxIndex = indexPairs[n]
            color = color_list[n]
            color1 = 'k'
            labelHeight = labelHeight_list[n]
    
            minPoint, maxPoint = image_tab[n][minIndex], image_tab[n][maxIndex]
    
            # up arrow
            ax.annotate('', xy = (maxPoint, emax), xytext = (maxPoint, emin),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = color, edgecolor = color,
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
    
            # down arrow
            ax.annotate('',
                        xy = (maxPoint, emin), xytext = (maxPoint, emax),
                        xycoords = 'data', textcoords = 'data',
                        arrowprops = dict(facecolor = color, edgecolor = color,
                        shrinkA = 0, shrinkB = 0, width = 1, headwidth = 8)) 
    
            # horizontal line connecting down arrow tip to energy minimum
#            ax.plot([minPoint, maxPoint], [emin, emin], ls = '--', color = color)
            
            if barriers:
    
                halign = 'left'
                valign = 'center'
                xpos = maxPoint + .3
                ypos = (emax - emin) * labelHeight + emin
        
                # label arrow with barrier value
                ax.text(xpos, ypos, '%.2f' %(kbarrier), fontsize = 15, color = color1,
                         verticalalignment = valign, horizontalalignment = halign)
    
    # plot energy profiles
    for n in range(len(en_tab)):
        color = color_list[n]
        en_list = en_tab[n]
        image_list  = image_tab[n]

        # plot profiles with dots and lines
        label = label_list[n] # I stupidly used 'n' as the number index
        ax.plot(image_list, en_list, 'o', color = color, mec = color)
        ax.plot(image_list, en_list, color = color, mec = color, linewidth = 2, label = label)

    # plot attributes
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.set_xlim(0, 100)
    if reference == 'end':
        ax.set_ylim(-.4, 1.9)
    if ebounds != None:
        ax.set_ylim(ebounds)
    ax.axhline(0, color = 'white', zorder = -3) # x-axis

    # white axes with no border
    for spine in ax.spines:
        ax.spines[spine].set_color('white')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # white axes and tick labels at select points
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis = 'x', colors = 'white', labelsize = 14) 
    ax.tick_params(axis = 'y', colors = 'white', labelsize = 14) 
#    ax.tick_params(axis = 'x', labelsize = 14) 
#    ax.tick_params(axis = 'y', labelsize = 14) 
    ax.set_xticks([0, 100])
    ax.set_yticks([0, 0.8])

    # save and show
    plt.tight_layout()
    if save:
        plt.savefig(outfile, transparent = True)
    plt.show()


#----------------------------- old. requires energy file -----------------------------------
def plotfromfile(
         infile = 'energies',
         figsize = (10, 6),
         ratio = (3, 1),
         show = True,
         save = False,
         outfile = 'barriers.pdf',
         tmd_list = ['mos$_2$', 'ws$_2$', 'mose$_2$', 'wse$_2$'],
         color_list = ['blue', 'green', 'red', 'orange'],
         resolution = 200,
         show_formation = False,
         ebounds = None,
         ):
    """
    plots energy barriers from separate energy file
    """
    # extract energies from infile
    energy_t2 = []
    with open(infile) as f:
        for line in f:
            energy_str = line.strip('\n').split('\t')
            energy_t1 = [float(energy) for energy in energy_str]
            energy_t2.append(energy_t1)

    images = arange(len(energy_t2))
    energy_t2 = transpose(energy_t2)
    domain = linspace(images[0], images[-1], resolution)

    # create figure
    fig = plt.figure(figsize = figsize)
    gs = gridspec.gridspec(1, ratio[0] + ratio[1])
    ax1 = fig.add_subplot(gs[0, :ratio[0]])   # profiles take up 2/3
    ax2 = fig.add_subplot(gs[0, -ratio[1]], sharey = ax1)
    for label in ax2.get_yticklabels():
        label.set_visible(false)       # don't renumber energy axis on bar plot
    fig.subplots_adjust(wspace = 0.05) # less space between subplots
    if ebounds != none:
        ax1.set_ylim(ebounds)

    # energy profile subplot attributes
    ax1.set_xlabel('reaction coordinate', fontsize = 18)
    ax1.set_ylabel('energy (ev)', fontsize = 18)
    ax1.set_xlim(images[0], images[-1])
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax1.axhline(0, color = 'black')
    ax1.grid()

    # energy barrier subplot attributes
    ax2.axhline(0, color = 'black')
    ax2.grid()
    width = .3
    num_tmds = len(tmd_list)
    ax2.set_xticks(arange(num_tmds) / 3)
    ax2.set_xticklabels(tmd_list, rotation = 'vertical', fontsize = 14)
    
    # energy barriers
    kbarriers = [max(energy_t2[spec]) for spec in range(num_tmds)]
    tbarriers = [energy_t2[spec][-1] for spec in range(num_tmds)]
    ax2.bar(arange(num_tmds) / 3, kbarriers, width, color = color_list, edgecolor = color_list, zorder = 2)
    if show_formation:
        ax2.bar(arange(num_tmds) / 3, tbarriers, width, color = "white", edgecolor = color_list, hatch = "//", zorder = 2)

    # plot
    for spec in range(num_tmds):
        color = color_list[spec]
        energies = energy_t2[spec]

        # energy profiles with cubic interpolation
        cs = interp1d(images, energies, kind = 'cubic')
        ax1.plot(images, energies, 'o', color = color, mec = color)
        ax1.plot(domain, cs(domain), linewidth = 2, color = color, label = tmd_list[spec])

        # show values of kbarriers
        ax2.text(spec / 3, kbarriers[spec] + .05,
                 '%.3g' %(kbarriers[spec]), fontsize = 12, ha = 'center')

        # show values for tbarriers
        offset = -.25
        color = 'black'
        if tbarriers[spec] > 0:
            offset = .05
            color = 'white'
        if show_formation:
            ax2.text(spec / 3, tbarriers[spec] + offset, str(round(tbarriers[spec], 2)),
                     fontsize = 12, ha = 'center', color = color)

    ax1.legend(loc = 2, fontsize = 14)
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    if show:
        plt.show()

#---------------------------- debugging ----------------------------------
alphabet = 'abcdefghijklmnopqrstuvwxyz'

from random import shuffle

def preptestdirectories(numletters = 16):
    """ prepares test directories for plotsegments() """
    alpha_list = [letter for letter in alphabet]

    start_list = alpha_list[:-1]
    start_list = ['a', 'a', 'a'] + start_list
    end_list = alpha_list[1:]
    end_list += ['z', 'z', 'z']
    shuffle(start_list)
    shuffle(end_list)
    for start, end in zip(start_list[:numletters], end_list[:numletters]):
        if start != end:
            directory = '%sto%s' %(start, end)
            for n in range(10):
                outfile = '%s/pa_e5_1/%.2d/outcar' %(directory, n)
                os.makedirs(os.path.dirname(outfile), exist_ok = true)
                with open(outfile,'w') as f:
                    f.write('test')
        else:
            pass
    
def istwo(num):
    if num == 2:
        return 1
    else:
        return 0

def gettwo(num):
    if num == 2:
        print('done')
        return 1
    elif num > 2:
        print('subtracting 1')
        num = gettwo(num - 1)
    elif num < 2:
        print('adding 1')
        gettwo(num + 1)

    return 1

#----------------------------------- scratch -------------------------------------
#    for directory in dir_list:
#        start, end = directory.split('to')
#        seg_list
#
#    # find which pairs of paths should be connected connect
#    start_dict = {} # start position : n
#    end_dict = {}   # end position   : m
#    joint_dict = {} # joint position : (m, n)
#    for n, directory in enumerate(dir_list):
#        start, end = directory.split('to')
#        start_dict[start] = n
#        end_dict[end] = n
#        if end in start_dict:
#            m = start_dict[end]
#            joint_dict[end] = (n, m)
#        if start in end_dict:
#            m = end_dict[start]
#            joint_dict[start] = (m, n)

#
    # separate MAPbI3 from BaZrS3 by background color
#    numBars = boundary_list[-1] + 1
#    z = -1
#    for system, boundary in zip(system_list, boundary_list):
#        fraction = (boundary + 1) / numBars
#        background = Rectangle((0, 0), width = fraction, height = 1, facecolor = color_list[z], transform = ax.transAxes)
#        ax.add_patch(background)
#        z -= 1

#    domain_tab = [[linspace(0, len(en_tab[0]) - 1, resolution)] for e in range(len(en_tab))]
#    tbarriers = [en_list[-1] for en_list in en_tab]
