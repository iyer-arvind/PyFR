# -*- coding: utf-8 -*-

#from pyfr.readers.native import read_pyfr_data
from pyfr.inifile import Inifile


class BaseWriter(object):
    """Functionality for post-processing PyFR data to visualisation formats"""

    def __init__(self,solution,divisor,outf,precision):
        """Loads PyFR mesh and solution files

        A check is made to ensure the solution was computed on the mesh.

        :param args: Command line arguments passed from scripts/postp.py
        :type args: class 'argparse.Namespace'

        """
        self.precision = precision
        self.divisor=divisor
        self.outf = outf

        self.soln = solution

        # Load config file
        self.cfg = Inifile(solution.getConfig())
