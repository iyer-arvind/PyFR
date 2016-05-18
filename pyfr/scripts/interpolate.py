#! /usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import FileType
from collections import OrderedDict
import re

#import mpi4py.rc
#mpi4py.rc.initialize = False

import h5py
import numpy as np
from scipy.spatial.ckdtree import cKDTree as KDTree

from pyfr.inifile import Inifile
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.solvers.base import BaseSystem
from pyfr.util import proxylist, subclass_where, subclasses


def add_args(ap_interpolate):

    ap_interpolate.add_argument('inmesh', type=str,
                                help='input mesh file')

    ap_interpolate.add_argument('insolution', type=str,
                                help='input solution file')

    ap_interpolate.add_argument('outmesh', type=str,
                                help='output PyFR mesh file')

    ap_interpolate.add_argument('outconfig', type=FileType('r'),
                                help='output config file')

    ap_interpolate.add_argument('outsolution', type=str,
                                help='output solution file')

    ap_interpolate.set_defaults(process=process_interpolate)


def get_eles(rallocs, mesh, soln, cfg):
        # Create a backend
        systemcls = subclass_where(BaseSystem,
                                   name=cfg.get('solver', 'system'))

        # Get the elementscls
        elementscls = systemcls.elementscls

        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Look for and load each element type from the mesh
        elemap = OrderedDict()
        for f in mesh:
            m = re.match('spt_(.+?)_p%d$' % rallocs.prank, f)
            if m:
                # Element type
                t = m.group(1)

                elemap[t] = elementscls(basismap[t], mesh[f], cfg)

        # Construct a proxylist to simplify collective operations
        eles = proxylist(elemap.values())

        # Process the solution
        if soln:
            for k, ele in elemap.items():
                soln = soln['soln_%s_p%d' % (k, rallocs.prank)]
                ele.set_ics_from_soln(soln, cfg)

        return eles


eps = (np.finfo('float').eps ** (1.0/3.0))


def inside_quad(x):
    return np.all(abs(x) <= 1+eps)


def inside_tri(x):
    return (x[0] >= -1 - eps) and (x[1] >= -1 - eps) and (x[0] + x[1]) <= eps


def inside_tet(x):
    return ((x[0] >= -1 - eps) and (x[1] >= -1 - eps) and (x[2] >= -1 - eps)
            and (x[0] + x[1] + x[2]) <= -1 + eps)


def inside_hex(x):
    return np.all(abs(x) <= (1 + eps))


def inside_prism(x):
    return (
        (x[0] >= -1 - eps) and
        (x[1] >= -1 - eps) and
        (x[0] + x[1]) <= eps
        and abs(x[2]) <= 1
    )


inside = dict(quad=inside_quad, tri=inside_tri, tet=inside_tet,
              hex=inside_hex, prism=inside_prism)


class PointLocator(object):
    def __init__(self, in_eles):
        self.in_eles = in_eles
        self._locator = {}

        for ele_typ in self.in_eles:
            el_list = np.rollaxis(ele_typ.eles, 1)
            el_cent = np.average(el_list, 1)
            self._locator[ele_typ.basis.name] = KDTree(el_cent)

    def __call__(self, xyl):
        in_eles = self.in_eles
        ndims = in_eles[0].ndims

        # The origin in the transformed space
        origin = np.array((0.0, 0.0, 1.0) if ndims == 2
                          else (0.0, 0.0, 0.0, 1.0))

        # Get a list of distances to the element centers to all elements
        n_neigh = 20
        ele_listp = np.zeros((len(in_eles), xyl.shape[0], n_neigh),
                             dtype=[('d', 'f'), ('idx', 'i'), ('typ', 'U4')])
        ele_map = {}

        if True:
            for e, ele_typ in enumerate(in_eles):
                ele_map[ele_typ.basis.name] = ele_typ
                d, idx = self._locator[ele_typ.basis.name].query(xyl, n_neigh)

                ele_listp['d'][e] = d
                ele_listp['idx'][e] = idx
                ele_listp['typ'][e] = ele_typ.basis.name

        ele_listp = np.vstack(ele_listp)

        ret = []
        pb = ProgressBar(0, 0, xyl.shape[0])
        for ii, (ele_list, xy) in enumerate(zip(ele_listp, xyl)):
            pb.advance_to(ii)

            # Sort by distance
            ele_list = np.sort(ele_list, order=['d'])

            xy = np.hstack([xy, (1, )])

            # Iterate from the closest element with increasing distance
            for d, idx, typ_name in ele_list:
                # Get the element
                ele_typ = ele_map[typ_name]
                el = ele_typ.eles[:, idx, :]

                # Start with a guess that the element is at the origin
                testp = np.array([-1-eps, eps, eps, 1])
                oldp = list()

                basis = ele_typ.basis.sbasis

                # We should get to the element within a few iterations
                for itrn in range(8):
                    trns = np.eye(ndims+1)
                    jac = np.squeeze(basis.jac_nodal_basis_at([testp[:-1]]))

                    # The translation of the element is the translation of
                    # the origin
                    nbs_o = basis.nodal_basis_at([origin[:-1]])
                    trns[:ndims, :ndims] = np.dot(jac, el).T
                    trns[:ndims, ndims] = np.dot(nbs_o, el)

                    newp = np.linalg.solve(trns, xy)

                    # If the new location is close to the previous, abort
                    if np.allclose(testp, newp, 1e-8, 1e-8):
                        break

                    # Else, need to update
                    oldp.append(testp)
                    testp = newp

                else:
                    raise ValueError('Locating iterations did not converge')

                # Check if this point is inside the domain
                if inside[typ_name](testp):
                    ret.append((ele_typ, idx, testp))
                    break
            else:
                # Could not find a suitable element
                raise ValueError(
                    'Could not locate point {} in domain'.format(xy))

        return ret


def process_interpolate(args):
    # Import MPI
    #from mpi4py import MPI

    # Manually initialise MPI
    #MPI.Init()

    # Read the input mesh
    in_mesh = NativeReader(args.inmesh)

    # Read the output mesh
    out_mesh = NativeReader(args.outmesh)

    # Read the in solution
    in_solution = NativeReader(args.insolution)

    # Read the in config from solution file
    in_cfg = Inifile(in_solution['config'])

    # Read the output config
    out_cfg = Inifile.load(args.outconfig)

    # Get in rallocs
    rallocs = get_rank_allocation(in_mesh, in_cfg)

    # Load the elements of the input mesh
    in_eles = get_eles(rallocs, in_mesh, in_solution, in_cfg)

    # Load the elements of the output mesh
    out_eles = get_eles(rallocs, out_mesh, None, out_cfg)

    locator = PointLocator(in_eles)

    # Create the solution map for output
    solnmap = OrderedDict()
    solnmap['mesh_uuid'] = out_mesh['mesh_uuid']
    solnmap['stats'] = in_solution['stats']
    solnmap['config'] = out_cfg.tostr()

    for oe in out_eles:
        # Set up the nodal basis for the shape points in the out mesh
        plocop = oe.basis.sbasis.nodal_basis_at(oe.basis.upts)

        # Get the locations of the shape points
        plocupts = np.dot(plocop, oe.eles.reshape(oe.nspts, -1))

        coords = plocupts.reshape(oe.nupts*oe.neles, oe.ndims)
        out_soln = np.empty((oe.nupts*oe.neles, oe.nvars))

        print('Interpolating to ', oe.basis.name)
        pb = ProgressBar(0, 0, coords.shape[0])

        for idxp, (xy, (ele_typ, idx, newp)) in \
                enumerate(zip(coords, locator(coords))):
            pb.advance_to(idxp)

            interp = ele_typ.basis.ubasis.nodal_basis_at([newp[:-1]])
            sol = np.dot(interp, ele_typ._scal_upts[:, :, idx])
            out_soln[idxp, :] = sol

        out_soln = out_soln.reshape((oe.nupts, oe.neles, oe.nvars))
        out_soln = np.swapaxes(out_soln, 1, 2)

        solnmap['soln_{0}_p0'.format(oe.basis.name)] = out_soln

    with h5py.File(args.outsolution, 'w') as msh5:
        for k, v in solnmap.items():
            msh5.create_dataset(k, data=v)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    add_args(parser)
    process_interpolate(parser.parse_args())
