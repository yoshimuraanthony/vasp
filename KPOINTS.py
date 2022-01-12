# Anthony Yoshimura
# 02/12/18

# KPOINTS class for quick and intuitive manipulations of KPOINTS in python

from numpy import array, zeros, append
from numpy import dot, tensordot, cross
from numpy import sqrt, sin, cos, arccos, radians, degrees, pi, floor, ceil, log10
from numpy.linalg import norm, inv
from subprocess import call 
from inspect import stack
from copy import deepcopy
from makeReadable import alignDecimals
import os
import periodic

point_dict = {
    '\Gamma': [0.0, 0.0, 0.0],
    'M'     : [0.5, 0,0, 0.0],
    'K'     : [1/3, 1/3, 0.0],
    }

label_dict = {
    (0.0, 0.0, 0.0): '\Gamma',
    (0.5, 0,0, 0.0): 'M',
    (1/3, 1/3, 0.0): 'K',
    }

def getKPOINTS(infile = 'KPOINTS'):
    """
    returns a list of lists representing a set of KPOINTS
    infile: KPOINTS file (str)
    """
    k_tab = []
    with open(infile) as f:
        for line in f:
            str_list = line.split()
            new_list = []
            for s in str_list:
                if s.isdigit():
                    s = int(s)
                else:
                    try:
                        s = float(s)
                    except ValueError:
                        pass
                new_list.append(s)
            if '#' not in new_list and len(new_list) > 0:
                k_tab.append(new_list)

    return k_tab

def makeKPOINTS(ktype = 'Line-mode', comment = 'band structure', number = 64):
    """
    returns KPOINTS instance
    """
    k_tab = [[comment], [number]]
    if ktype[0] == 'l' or ktype[0] == 'L':
        k_tab.append([ktype])
        k_tab.append(['reciprocal'])

    else:
        k_tab.append([ktype])
        k_tab.append([1, 1, 1])
        
    return KPOINTS(k_tab)
        

class KPOINTS(object):
    """
    mutable representation of KPOINTS as a list of lists
    """
    # UNDER CONSTRUCTION: mesh type KPOINTS

    def __init__(self, k_tab = 'KPOINTS'):
        """
        Imports KPOINTS data from either a file or a table
        k_tab: KPOINTS file or table (str or list of lists)
        """
        if type(k_tab) == list:
            self.k_tab = k_tab
        elif type(k_tab) == str:
            self.k_tab = getKPOINTS(k_tab)
        else:
            print('table must be a file or list of lists')

        # add [:] to avoid aliasing
        self.comment = self.k_tab[:1][0][:]
        self.number = self.k_tab[1:2][0][0]
        self.type = self.k_tab[2:3][0][0]
        if self.type[0] == 'g' or self.type[0] == 'G' or\
           self.type[0] == 'm' or self.type[0] == 'M' or\
           self.type[0] == 'a' or self.type[0] == 'A':
            self.mesh = self.k_tab[3:4][0][:] 
            self.shift = self.k_tab[4:5][0][:]
        elif self.type[0] == 'l' or self.type[0] == 'L':
            self.lines = [line[:3] for line in self.k_tab[4:]]
            self.labels = [line[-1] for line in self.k_tab[4:]]

    def write(self, outfile = 'KPOINTS_scr'):
        """
        writes self.k_tab to outfile
        outfile: file to which self.k_tab is written (str)
        """
        # write path if it doesn't already exist
        if '/' in outfile:
            os.makedirs(os.path.dirname(outfile), exist_ok = True)

        # combine all elements into k_tab
        self.updateTable()

        # write to outfile
        with open(outfile, 'w') as f:
            
            # comment, number, type, and either mesh or 'reciprocal'
            for row in self.k_tab[:4]:
                for element in row:
                    f.write(str(element) + ' ')
                f.write('\n')

            # lines
            if self.type[0] == 'l' or self.type[0] == 'L':
                alignedPoints = alignDecimals(self.lines, 8)
                for ind, (point, label) in enumerate(zip(alignedPoints, self.labels)):
                    f.write('   ')
                    for comp in point:
                        f.write(str(comp) + ' ')
                    f.write('! ' + label)
                    f.write('\n')
                    if ind % 2 == 1:  # vertical space between pairs of points
                        f.write('\n')

            # shift (for mesh type)
            else:
                for element in self.k_tab[4]:
                    f.write(str(element) + ' ')

    def addLines(self, point_list):
        """
        adds line to self.line
        line_list: list of end points
            * list of coordinate pairs (tuples of three floats)
            * list of symmetry-point pairs
        """
        lineNumber = int(len(self.lines)/2 + 1)
        for ind, point in enumerate(point_list):

            if type(point) == str:
                label = point
                point = point_dict[label]
            else:
                if ind % 2 == 0:
                    label = ('start line %s' %lineNumber)            
                else: 
                    label = ('end line %s' %lineNumber)            
                    lineNumber += 1
            self.labels.append(label)
            self.lines.append(point)

    def copy(self):
        """ returns identical KPOINTS class """
        if stack()[1][3] == '<module>':
            self.updateTable()

        self.updateTable()

        return KPOINTS(self.k_tab)
        
    def prepFoldedBS(self, dim):
        """
        prepares KPOINTS files to produce folded band structures plot
        dim: supercell dimenssion (list of three pos ints)
        """
        # UNDER CONSTRUCTION: giving me
        #     ValueError: operands could not be broadcast together with shapes (0,) (3,)
        dim1, dim2, dim3 = dim
        in1, in2, in3 = 1/dim[0], 1/dim[1], 1/dim[2]
        for i in range(dim1):
            for j in range(dim2):
                for k in range(dim3):
                    translation = [i*in1, j*in2, k*in3]
                    ShiftedKPOINTS = self.copy()
                    ShiftedKPOINTS.scale([in1, in2, in3])
                    ShiftedKPOINTS.translate(translation)
                    directory = 'fold%s%s%s' %(i, j, k)
                    ShiftedKPOINTS.write('%s/KPOINTS' %directory)

    def scale(self, scale, line_list = 'all'):
        """
        scale specified lines in specified directions
        scale: vector by which all lines points are muliplied (float)
        line_list: lines to be scale (list of pos ints)
            * if 'all', translates all lines
        """
        # UNDER CONSTRUCTION: scale specified lines only
        if stack()[1][3] == '<module>':
            self.updateTable()

        if line_list == 'all':
            self.lines = (array(deepcopy(self.lines)) * array(scale)).tolist()

    def translate(self, translation, line_list = 'all'):
        """
        translates specified lines in specified directions
        translation: translation vector (list of 3 floats)
        line_list: lines to be translated (list of pos ints)
            * if 'all', translates all lines
        """
        # UNDER CONSTRUCTION: translate specified lines only
        if stack()[1][3] == '<module>':
            self.updateTable()

        if line_list == 'all':
            self.lines = (array(deepcopy(self.lines)) + translation).tolist()

    def undo(self):
        """ Undoes last change """
        self.__init__(self.k_tab)

    def updateTable(self):
        """ Constructs table with up-to-date attributes """
        # line type
        if self.type[0] == 'l' or self.type[0] == 'L':
            self.k_tab = [self.comment[:]] + [[self.number]] \
                + [[self.type]] + [['reciprocal']]
            for point, label in zip(self.lines, self.labels):
                self.k_tab.append(point + ['!'] + [label])

        # mesh type
        else:
            self.k_tab = [self.comment[:]] + [[self.number]] \
                + [[self.type]] + [self.mesh[:]] \
                + [self.shift[:]]

                
