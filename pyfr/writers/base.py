# -*- coding: utf-8 -*-

from abc import abstractmethod

from pyfr.readers.native import read_pyfr_data
from pyfr.solvers import BaseSystem
from pyfr.util import subclass_where


class BaseWriter(object):
    """Functionality for post-processing PyFR data to visualisation formats"""

    def __init__(self, args, cfg):
        """Loads PyFR mesh and solution files

        A check is made to ensure the solution was computed on the mesh.

        :param args: Command line arguments passed from scripts/postp.py
        :type args: class 'argparse.Namespace'

        """
        # Load mesh
        self.mesh = read_pyfr_data(args.meshf)

        # Mesh info
        self.mesh_inf = self.mesh.array_info

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]

        # Save the config
        self.cfg = cfg

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls

    @abstractmethod
    def write_out(self, file_name, soln):
        pass
