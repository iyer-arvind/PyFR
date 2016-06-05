# -*- coding: utf-8 -*-

import itertools as it
import os

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.nputil import fuzzysort
from pyfr.mpiutil import get_comm_rank_root


class LineShape(BaseShape):
    name = 'line'
    ndims = 1

    npts_coeffs = [1, 0]
    npts_cdenom = 1

    faces = []

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)
        return list(p[::-1] for p in it.product(pts1d, repeat=cls.ndims))


class SpatialAverage(BasePlugin):
    name = 'spatial_average'
    systems = ['*']
    swaps = {'xz': (1, 0, 2)}

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.directions = self.cfg.get(cfgsect, 'directions')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        self.basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        self.basename = self.cfg.get(cfgsect, 'basename')

        # Append the relevant extension
        if not self.basename.endswith('.csv'):
            self.basename += '.csv'

        assert len(intg.system.ele_map) == 1

        # Locations of the shape points  nspts x neles x ndims
        # Roll to have neles x nspts x ndims
        self.spts = np.swapaxes(
                [a.eles for a in intg.system.ele_map.values()][0], 0, 1)

        # Number of shape points per element
        n_eles, n_spts, n_dims = self.spts.shape

        # Locations of the solution points  nupts x ndims x neles
        # Rotate to have neles x ndims x nupts
        self.ploc_upts = np.swapaxes(intg.system.ele_ploc_upts[0], 0, 2)

        # Basis classes
        self.basis = [a.basis for a in intg.system.ele_map.values()][0]

        cfg = Inifile(intg.cfg.tostr())
        # We need both the 2d and the 1d elements so add the data in cfg
        cfg.set('solver-elements-quad', 'soln-pts',
                intg.cfg.get('solver-elements-hex', 'soln-pts'))

        cfg.set('solver-elements-line', 'soln-pts',
                intg.cfg.get('solver-elements-hex', 'soln-pts'))

        # The 2d and 1d basis class
        l_basiscls = subclass_where(BaseShape, name='line')(
            {8: 2}[n_spts], cfg
        )

        p_basiscls = subclass_where(BaseShape, name='quad')

        # The 2d and 1d basis class
        l_basiscls = subclass_where(BaseShape, name='line')(
            {8: 2}[n_spts], cfg
        )
        p_basiscls = subclass_where(BaseShape, name='quad')

        # The instance of the plane class
        self.plane = p_basiscls({8: 4}[n_spts], cfg)

        # The number of solution points in the plane and the line
        self.n_upts_line = n_upts_line = l_basiscls.upts.shape[0]
        n_upts_plane = self.plane.upts.shape[0]

        n_upts = self.basis.upts.shape[0]

        self.splits = np.cumsum([n_upts_plane
                                 for _ in range(n_upts_line)])[:-1]

        self.idx = []
        # Iterate over each element, p:plocs, s: solution
        for i, p in enumerate(self.ploc_upts):
            # Rotate the plocs such that the non-homogeneous durection
            # is the first index
            c = p[self.swaps[self.directions], ...]

            # Get the sorted order of elements wrt direction
            self.idx.append(fuzzysort(c, range(n_upts)))

        self.first_iteration = True
        self.nout = 0

    def _get_output_path(self, tcurr):
        # Substitute {t} and {n} for the current time and output number
        fname = self.basename.format(t=tcurr, n=self.nout)

        return os.path.join(self.basedir, fname)

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps != 0:
            return

        soln = np.swapaxes(intg.soln[0], 0, 2)
        n_eles, n_vars, n_upts = soln.shape

        line_sol = np.zeros((n_eles, self.n_upts_line, n_vars))

        if self.first_iteration:
            # Set the lineplocs
            line_ploc_upts = np.zeros((n_eles, self.n_upts_line))

        for i, (idx, p, s) in enumerate(zip(self.idx, self.ploc_upts, soln)):
            # Split the arrays after reordering them, now each split
            # has the plocs of the homogeneous direction plane
            # and the homogeneous direction solution.
            for pi, (pse, sse) in enumerate(
                    zip(np.split(p[:, idx], self.splits, 1),
                        np.split(s[:, idx], self.splits, 1))
            ):
                # Actual ploc of the non-homogeneous direction
                hd = pse[self.swaps[self.directions][0]]

                # Uncomment to enable check
                assert np.max(np.abs(hd - hd[0])) < 1e-8

                # Integrate the variables and divide by the std-element
                # area, which is 4
                vals = np.dot(sse, self.plane.upts_wts) / 4

                if self.first_iteration:
                    # Set the lineplocs
                    line_ploc_upts[i, pi] = hd[0]

                # Set the solution
                line_sol[i, pi, :] = vals

        # MPI info
        comm, rank, root = get_comm_rank_root()
        if self.first_iteration:
            line_ploc_upts_all = comm.gather(line_ploc_upts.flatten())
            if rank == root:
                line_ploc_upts_all = np.concatenate(line_ploc_upts_all)

                # Get the order of sorting
                ordering = np.argsort(line_ploc_upts_all)
                pc = None
                self.lists = lists = []
                for c, o in zip(line_ploc_upts_all[ordering], ordering):
                    if pc is None or abs(c - pc) > 1e-8:
                        lists.append((c, []))
                    lists[-1][1].append(o)
                    pc = c

        line_sols = comm.gather(line_sol.reshape([-1, n_vars]))
        if rank == root:
            line_sols = np.concatenate(line_sols)
            a = [(c, np.average(line_sols[l, :], axis=0))
                 for c, l in self.lists]
            coords, data = zip(*a)
            data = np.hstack((np.array(coords)[:, np.newaxis], data))
            np.savetxt(self._get_output_path(intg.tcurr), data)
            self.nout += 1

        self.first_iteration = False
