#!/usr/bin/env python

import itertools as it
import re

import numpy as np

from pyfr.readers.native import NativeReader
from pyfr.inifile import Inifile
from pyfr.nputil import fuzzysort
from pyfr.shapes import BaseShape
from pyfr.solvers.navstokes import NavierStokesSystem
from pyfr.util import subclass_where

d_rules = {'hex': {'yz': ('quad', [()])}}


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


class SpatialAverage(object):
    def __init__(self, mesh_file, directions):

        # Load mesh and solution files
        self.mesh = NativeReader(mesh_file)

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info

        self.directions = directions

    def average(self, soln):
        # Readers and the ini file
        soln = NativeReader(soln)
        cfg = Inifile(soln['config'])

        swaps = {'xz': (1, 0, 2)}

        # Perpendicular coordinate and solution will be listed here
        line_sols = ([], [])

        # Iterate over the mesh
        for m_name in self.mesh:
            # Search only for shape points
            m = re.match('spt_' + r'([a-z]+)_p(\d+)', m_name)
            if not m:
                continue

            print(m_name)

            # Get the shape name and the partition
            shp, part = m.groups()
            part = int(part)

            # Get the shape pts
            spts = self.mesh[m_name]

            # Number of shape points
            n_spts = spts.shape[0]

            # This code works only with hex
            if shp not in ('hex', ):
                raise ValueError

            # Get the corresponding solution file
            for s_name in soln:
                m = re.match(r'([a-zA-Z_]+)_{}_p{}'.format(shp, part), s_name)
                if m:
                    break

            else:
                raise ValueError(
                    'Could not find solution for {}'.format(m_name)
                )

            # Get the basis cls for the element
            basiscls = subclass_where(BaseShape, name=shp)

            # We need both the 2d and the 1d elements so add the data in cfg
            cfg.set('solver-elements-quad', 'soln-pts',
                    cfg.get('solver-elements-hex', 'soln-pts'))

            cfg.set('solver-elements-line', 'soln-pts',
                    cfg.get('solver-elements-hex', 'soln-pts'))

            # The 2d and 1d basis class
            l_basiscls = subclass_where(BaseShape, name='line')({8: 2}[n_spts], cfg)
            p_basiscls = subclass_where(BaseShape, name='quad')

            # The instance of the plane class
            plane = p_basiscls({8: 4}[n_spts], cfg)

            # The number of solution points in the plane and the line
            n_upts_line = l_basiscls.upts.shape[0]
            n_upts_plane = plane.upts.shape[0]

            # Get the solution
            sol = soln[s_name].swapaxes(0, 2)

            # The number of elements, vars, and solution points
            n_eles, n_vars, n_upts = sol.shape

            # This will accumulate the plane averaged solutions
            linesol = np.zeros((n_eles, n_upts_line, n_vars))

            # This will accumulate the coordinate of the plane
            lineupts = np.zeros((n_eles, n_upts_line))

            # The solution will be split plane-wise into n_upts_line parts
            # and each part will have n_upts_plane solution points
            splits = np.cumsum([n_upts_plane for _ in range(n_upts_line)])[:-1]

            # We need the ploc_upts to decide the order, this needs the element
            ele = NavierStokesSystem.elementscls(basiscls, spts, cfg)

            ploc_upts = ele.ploc_at_np('upts').swapaxes(0, 2)

            # Iterate over each element, p:plocs, s: solution
            for i, (p, s) in enumerate(zip(ploc_upts, sol)):
                # Rotate the plocs such that the non-homogeneous durection
                # is the first index
                c = p[swaps[self.directions], ...]

                # Get the sorted order of elements wrt direction
                idx = fuzzysort(c, range(n_upts))

                # Split the arrays after reordering them, now each split
                # has the plocs of the homogeneous direction plane
                # and the homogeneous direction solution.
                for pi, (pse, sse) in enumerate(
                        zip(np.split(p[:, idx], splits, 1),
                            np.split(s[:, idx], splits, 1))
                ):
                    # Actual ploc of the non-homogeneous direction
                    hd = pse[swaps[self.directions][0]]

                    # Uncomment to enable check
                    # assert np.max(np.abs(hd - hd[0])) < 1e-8

                    # Integrate the variables and divide by the std-element
                    # area, which is 4
                    vals = np.dot(sse, plane.upts_wts) / 4

                    # Set the ploc
                    lineupts[i, pi] = hd[0]

                    # Set the solution
                    linesol[i, pi, :] = vals

            # Flatten the plocs and the averaged solution and append
            line_sols[0].append(lineupts.flatten())
            line_sols[1].append(linesol.reshape([-1, n_vars]))

        # Assembling of the parts
        # First concatenate all the arrays from various partitions
        locs = np.concatenate(line_sols[0])
        sols = np.concatenate(line_sols[1])

        # Get the order of sorting
        idx = np.argsort(locs)

        # Next separate all the elements with similar value of the ploc
        # Within a fixed tolerance
        si, li = sols[idx[0], ...], locs[idx[0]]
        accm = [si]
        avg_loc = []
        avg_sol = []

        lidx = len(locs)
        for i, (sj, lj) in enumerate(zip(sols[idx[1:], ...], locs[idx[1:]]),
                                     start=1):
            # If the next ploc is close to the current one, just append
            if i < lidx-1 and abs(li-lj) < 1e-5:
                accm.append(sj)
                continue

            # Else save the ploc, and the average of all the solutions
            avg_loc.append(li)
            avg_sol.append(np.average(np.array(accm), 0))

            # Reset
            li = lj
            accm = [sj]

        # Finally concatenate the plocs with the solution
        avg_loc = np.array(avg_loc).reshape([-1, 1])
        avg_sol = np.array(avg_sol)

        # Return
        return np.concatenate((avg_loc, avg_sol), axis=1)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--mesh', type=str, required=True,
                        help='Mesh file')
    parser.add_argument('-d', '--direction', type=str, required=True,
                        help='Direction to average')
    parser.add_argument('-s', '--solution', type=str, nargs='+',
                        help='Solution file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output CSV file')

    args = parser.parse_args()
    averager = SpatialAverage(args.mesh, args.direction)
    avg = averager.average(args.solution[0])
    np.savetxt(args.output, avg)


if __name__ == '__main__':
    main()
