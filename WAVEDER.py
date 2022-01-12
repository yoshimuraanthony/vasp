#!/usr/bin/env python

# Anthony Yoshimura
# taken directly from pymatgen/io/vasp/outputs.py

import numpy as np

from copy import deepcopy
from inspect import stack
import os
from subprocess import call
from time import time

import matplotlib.pyplot as plt

from periodic import table as p_dict
import POSCAR as p

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings

warnings.filterwarnings('error')


class Waveder:
    """
    Class for reading a WAVEDER file.
    The LOPTICS tag produces a WAVEDER file.
    The WAVEDER contains the derivative of the orbitals with respect to k.
    Author: Kamal Choudhary, NIST

    Args:
        filename: Name of file containing WAVEDER.
    """

    def __init__(self, filename, gamma_only = False):
        with open(filename, 'rb') as fp:
            def readData(dtype):
                """ Read records from Fortran binary file and convert to
                np.array of given dtype. """
                data = b''
                while 1:
                    prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
                    data += fp.read(abs(prefix))
                    suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
                    if abs(prefix) - abs(suffix):
                        raise RuntimeError(f"""Read wrong amount of bytes.
                        Expected: {prefix:d}, read: {len(data):d}, suffix: {suffix:d}.""")
                    if prefix > 0:
                        break
                return np.frombuffer(data, dtype=dtype)

            nbands, nelect, nk, ispin = readData(np.int32)

            nodes_in_dielectric_function = readData(np.float)

            wplasmon = readData(np.float)

            if gamma_only:
                cder = readData(np.float)
            else:
                cder = readData(np.complex64)

            cder_data = cder.reshape((3, ispin, nk, nelect, nbands)).T

            self._cder_data = cder_data
            self._nkpoints = nk
            self._ispin = ispin
            self._nelect = nelect
            self._nbands = nbands

    @property
    def cder_data(self):
        """
        Returns the orbital derivative between states
        """
        return self._cder_data

    @property
    def nbands(self):
        """
        Returns the number of bands in the calculation
        """
        return self._nbands

    @property
    def nkpoints(self):
        """
        Returns the number of k-points in the calculation
        """
        return self._nkpoints

    @property
    def nelect(self):
        """
        Returns the number of electrons in the calculation
        """
        return self._nelect

    def get_orbital_derivative_between_states(self, band_i, band_j, kpoint, spin, cart_dir):
        """
        Method returning a value
        between bands band_i and band_j for k-point index, spin-channel and cartesian direction.
        Args:
            band_i (Integer): Index of band i
            band_j (Integer): Index of band j
            kpoint (Integer): Index of k-point
            spin   (Integer): Index of spin-channel (0 or 1)
            cart_dir (Integer): Index of cartesian direction (0,1,2)

        Returns:
            a float value
        """
        if band_i < 0 or band_i > self.nbands - 1 or band_j < 0 or band_j > self.nelect - 1:
            raise ValueError("Band index out of bounds")
        if kpoint > self.nkpoints:
            print('kpoint = %s, nkpts = %s' %(kpoint, self.nkpoints))
            raise ValueError("K-point index out of bounds")
        if cart_dir > 2 or cart_dir < 0:
            raise ValueError("cart_dir index out of bounds")

        return self._cder_data[band_i, band_j, kpoint, spin, cart_dir]


#from numpy import array, zeros, append, floor, ceil, arange, linspace
#from numpy import dot, tensordot, cross, sign
#from numpy import sqrt, sin, cos, arccos, radians, degrees, log10
#from numpy import e, pi
#from numpy.linalg import norm, inv
