# -*- coding: utf-8 -*-

import itertools as it
import os
import sys

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.nputil import fuzzysort, npeval
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

        self.basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        self.basename = self.cfg.get(cfgsect, 'basename')
        self.fpdtype = intg.backend.fpdtype

        # Expressions to time average
        c = self.cfg.items_as('constants', float)
        self.exprs = [(k, self.cfg.getexpr(cfgsect, k, subs=c))
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('avg-')]

        # Append the relevant extension
        if not self.basename.endswith('.csv'):
            self.basename += '.csv'

        # Output time step and next output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr + self.dt_out

        if self.dt_out:
            intg.call_plugin_dt(self.dt_out)

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

        self.elementscls = intg.system.elementscls

        # The 2d and 1d basis class
        self.line = subclass_where(BaseShape, name='line')(
            {8: 2}[n_spts], cfg
        )
        p_basiscls = subclass_where(BaseShape, name='quad')

        # The instance of the plane class
        self.plane = p_basiscls({8: 4}[n_spts], cfg)

        # The number of solution points in the plane and the line
        self.n_upts_line = n_upts_line = self.line.upts.shape[0]
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

    def _eval_exprs(self, soln, tcurr):
        exprs = []

        # Get the primitive variable names and solutions
        pnames = self.elementscls.privarmap[self.ndims]
        psolns = self.elementscls.con_to_pri(soln, self.cfg)

        # Prepare the substitutions dictionary
        ploc = dict(zip('xyz', self.ploc_upts.swapaxes(0, 1)))
        subs = dict(zip(pnames, psolns), t=tcurr, **ploc)

        # Evaluate the expressions
        exprs.append([npeval(v, subs) for k, v in self.exprs])

        # Stack up the expressions for each element type and return
        # exprs comes as n_vars x n_eles x n_upts
        return np.dstack(exprs)

    def _get_output_path(self, tcurr):

        # Substitute {t} and {n} for the current time and output number
        fname = self.basename.format(t=tcurr, n=self.nout)

        return os.path.join(self.basedir, fname)

    def __call__(self, intg):
        if abs(self.tout_next - intg.tcurr) > self.tol:
            return

        # soln comes as n_upts, n_vars, n_eles
        # getting to n_vars, n_eles, n_upts
        soln = np.swapaxes(intg.soln[0], 0, 1)
        self._run(soln, intg.tcurr, True)

        self.tout_next = intg.tcurr + self.dt_out

    def _run(self, soln, tcurr, write):
        # soln comes as n_vars x n_upts x n_eles
        exprs = self._eval_exprs(soln, tcurr)
        
        # exprs comes as n_vars x n_upts x n_eles
        # convert to n_eles x n_vars x n_upts
        exprs = np.rollaxis(exprs, 2, 0)
        n_eles, n_vars, n_upts = exprs.shape

        line_sol = np.zeros((n_eles, self.n_upts_line, n_vars), dtype=self.fpdtype)

        if self.first_iteration:
            # Set the lineplocs
            line_ploc_upts = np.zeros((n_eles, self.n_upts_line), dtype=self.fpdtype)

        for i, (idx, p, s) in enumerate(zip(self.idx, self.ploc_upts, exprs)):
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
            line_sol_shapes = comm.gather((rank, line_sol.shape))
            line_ploc_upts_all = comm.gather(line_ploc_upts.flatten())

            if rank == root:
                # Prepare the buffers for persistent communication
                line_ploc_upts_all = np.concatenate(line_ploc_upts_all)
                self._mpi_rbufs = mpi_rbufs = []
                self._mpi_rreqs = mpi_rreqs = []

                for rrank, shape in line_sol_shapes:
                    rbuf = np.empty(shape, dtype=self.fpdtype)
                    mpi_rbufs.append(rbuf)

                    if rrank != 0:
                        rreq = comm.Recv_init(rbuf, rrank, 0)
                        mpi_rreqs.append(rreq)

                # Get the order of sorting
                ordering = np.argsort(line_ploc_upts_all)
                pc = None
                self.lists = lists = []
                for c, o in zip(line_ploc_upts_all[ordering], ordering):
                    if pc is None or abs(c - pc) > 1e-8:
                        lists.append((c, []))
                    lists[-1][1].append(o)
                    pc = c

        u_bulk = 0

        from mpi4py import MPI

        if rank == root:
            # Copy over the local data
            self._mpi_rbufs[root][...] = line_sol

            # Wait for the remote data
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)
            line_sols = [l.reshape([-1, n_vars]) for l in self._mpi_rbufs]

            line_sols = np.concatenate(line_sols)
            a = [(c, np.average(line_sols[l, :], axis=0))
                 for c, l in self.lists]
            coords, data = zip(*a)

            if self.first_iteration:
                n_upts = self.line.upts.shape[0]

                crd = np.array(coords)
                n_eles = int(crd.shape[0]/self.line.upts.shape[0])

                cent = crd.reshape([n_eles, -1]).mean(axis=1)
                el_length = (crd - np.repeat(cent, n_upts)
                             ).reshape([-1, n_upts])[:, 0]/self.line.upts[0]

                self.wts = (np.tile(self.line.upts_wts, n_eles) *
                            np.repeat(el_length, n_upts))

                self.h = np.dot(np.zeros([n_eles*n_upts])+1, self.wts)

            c = np.array(coords)
            data = np.array(data)
            dc = c[1:] - c[:-1]
            count = len(np.where(dc>0.5)[0])+1
            u_idx = [i for i, e in enumerate(self.exprs) if e[0] == 'avg-u'][0]
            u_bulk = np.dot(data[:, u_idx], self.wts)/self.h/count

            data = np.hstack((np.array(coords)[:, np.newaxis], data))
            header = 'y ' + ' '.join([k.replace('avg-', '')
                                      for k, e in self.exprs])

            if write:
                np.savetxt(self._get_output_path(tcurr), data, header=header)
                self.nout += 1
        else:
            # Send data to root
            comm.Send(line_sol, root, 0)


        self.first_iteration = False
        return u_bulk
