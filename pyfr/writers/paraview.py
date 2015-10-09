# -*- coding: utf-8 -*-

"""Converts .pyfr[m, s] files to a Paraview VTK UnstructuredGrid File"""

from collections import defaultdict, OrderedDict
import os

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, register_finalize_handler
from pyfr.nputil import block_diag
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter


class ParaviewWriter(BaseWriter):
    # Supported file types and extensions
    name = 'paraview'
    extn = ['.vtu', '.pvtu']

    def __init__(self, args, cfg):
        super().__init__(args, cfg)

        self.dtype = np.dtype(args.precision).type
        self.divisor = args.divisor or self.cfg.getint('solver', 'order')

        # Import but do not initialise MPI
        from mpi4py import MPI

        # Manually initialise MPI
        MPI.Init()
        register_finalize_handler()
        comm, rank, root = get_comm_rank_root()

        nparts = len(self.mesh_inf)

        self.parallel_write = nparts > 1 and comm.size > 1

        self.outvarmap = list(self.elementscls.privarmap[self.ndims])
        self.visvarmap = self.elementscls.visvarmap[self.ndims]

        if self.export_gradients:
            if self.ndims == 2:
                self.outvarmap.extend(['rho_x', 'rhou_x', 'rhov_x', 'E_x',
                                       'rho_y', 'rhou_y', 'rhov_y', 'E_y'])

                self.visvarmap.update({
                    'grad_density': ['rho_x', 'rho_y'],
                    'grad_momentum': ['rhou_x', 'rhou_y', 'rhov_x', 'rhov_y'],
                    'grad_energy': ['E_x', 'E_y']})
            else:
                self.outvarmap.extend(['rho_x', 'rhou_x', 'rhov_x', 'rhow_x',
                                       'E_x', 'rho_y', 'rhou_y', 'rhov_y',
                                       'rhow_y', 'E_y', 'rho_z', 'rhou_z',
                                       'rhov_z', 'rhow_z', 'E_z'])

                self.visvarmap.update({'grad_density': ['rho_x', 'rho_y',
                                                        'rho_z'],
                                       'grad_momentum': ['rhou_x', 'rhou_y',
                                                         'rhou_z', 'rhov_x',
                                                         'rhov_y', 'rhov_z',
                                                         'rhow_x', 'rhow_y',
                                                         'rhow_z'],
                                       'grad_energy': ['E_x', 'E_y', 'E_z']
                                       })

        self._process_mesh()

    def _process_mesh(self):
        comm, rank, root = get_comm_rank_root()

        self.nsvpts = {}
        self.soln_vtu_op = {}
        self.vpts = {}
        self.vtu = {}

        for mk in self.mesh_inf:
            part_num = int(mk.rsplit('_', 1)[1][1:])
            if self.parallel_write and (part_num % comm.size) != rank:
                continue
            print('Mesh {} on {}'.format(mk, rank))
            name = self.mesh_inf[mk][0]

            mesh = self.mesh[mk]

            # Dimensions
            nspts, neles = mesh.shape[:2]

            # Get the shape and sub division classes
            shapecls = subclass_where(BaseShape, name=name)
            subdvcls = subclass_where(BaseShapeSubDiv, name=name)

            # Sub division points inside of a standard element
            svpts = shapecls.std_ele(self.divisor)
            self.nsvpts[mk] = nsvpts = len(svpts)

            # Shape
            soln_b = shapecls(nspts, self.cfg)

            # Generate the operator matrices
            mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
            self.soln_vtu_op[mk] = soln_b.ubasis.nodal_basis_at(svpts)

            # Calculate node locations of vtu elements
            vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
            vpts = vpts.reshape(nsvpts, -1, self.ndims)

            # Append dummy z dimension for points in 2D
            if self.ndims == 2:
                vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

            self.vpts[mk] = vpts

            # Perform the sub division
            nodes = subdvcls.subnodes(self.divisor)

            # Prepare vtu cell arrays
            vtu_con = np.tile(nodes, (neles, 1))
            vtu_con += (np.arange(neles)*nsvpts)[:, None]

            # Generate offset into the connectivity array
            vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
            vtu_off += (np.arange(neles)*len(nodes))[:, None]

            # Tile vtu cell type numbers
            vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)

            self.vtu[mk] = (vtu_con, vtu_off, vtu_typ)

    def _get_npts_ncells_nnodes(self, mk):
        m_inf = self.mesh_inf[mk]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=m_inf[0])
        subdvcls = subclass_where(BaseShapeSubDiv, name=m_inf[0])

        # Number of vis points
        npts = shapecls.nspts_from_order(self.divisor + 1)*m_inf[1][1]

        # Number of sub cells and nodes
        ncells = len(subdvcls.subcells(self.divisor))*m_inf[1][1]
        nnodes = len(subdvcls.subnodes(self.divisor))*m_inf[1][1]

        return npts, ncells, nnodes

    def _get_array_attrs(self, mk=None):
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize

        ndims = self.ndims
        vvars = OrderedDict(sorted(self.visvarmap.items(), key=lambda t: t[0]))

        names = ['', 'connectivity', 'offsets', 'types']
        types = [dtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        for fname, varnames in vvars.items():
            names.append(fname.capitalize())
            types.append(dtype)
            comps.append(str(len(varnames)))

        # If a mesh has been given the compute the sizes
        if mk:
            npts, ncells, nnodes = self._get_npts_ncells_nnodes(mk)
            nb = npts*dsize

            sizes = [3*nb, 4*nnodes, 4*ncells, ncells]
            sizes.extend(len(varnames)*nb for varnames in vvars.values())

            return names, types, comps, sizes
        else:
            return names, types, comps

    def write_out(self, file_name, soln):
        comm, rank, root = get_comm_rank_root()

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != soln['mesh_uuid']:
            raise RuntimeError('Solution was not computed on mesh')

        name, extn = os.path.splitext(file_name)

        parallel = extn == '.pvtu'
        assert not self.parallel_write or parallel, \
            'Cannot write vtu in parallel'

        parts = defaultdict(list)

        soln_inf = soln.array_info

        for mk, sk in zip(self.mesh_inf, soln_inf):
            part_num = int(mk.rsplit('_', 1)[1][1:])
            if self.parallel_write and (part_num % comm.size) != rank:
                continue
            prt = mk.split('_')[-1]
            pfn = '{0}_{1}.vtu'.format(name, prt) if parallel else file_name

            parts[pfn].append((mk, sk))

        write_s_to_fh = lambda s: fh.write(s.encode('utf-8'))

        for pfn, misil in parts.items():
            print('Soln {} on {}'.format(pfn, rank))

            with open(pfn, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              'version="0.1">\n<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0

                # Header
                for mk, sk in misil:
                    off = self._write_serial_header(fh, mk, off)

                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')

                # Data
                for mk, sk in misil:
                    self._write_data(fh, mk, soln[sk])

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        if parallel:
            if self.parallel_write:
                parts = comm.gather(tuple(parts.keys()), root=root)
                if rank != root:
                    return
                parts = [j for i in parts for j in i]

            with open(file_name, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="PUnstructuredGrid" '
                              'version="0.1">\n<PUnstructuredGrid>\n')

                # Header
                self._write_parallel_header(fh)

                # Constituent pieces
                for pfn in parts:
                    write_s_to_fh('<Piece Source="{0}"/>\n'
                                  .format(os.path.basename(pfn)))

                write_s_to_fh('</PUnstructuredGrid>\n</VTKFile>\n')

    @staticmethod
    def _write_darray(array, vtuf, dtype):
        array = array.astype(dtype)

        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _write_serial_header(self, vtuf, mk, off):
        names, types, comps, sizes = self._get_array_attrs(mk)
        npts, ncells = self._get_npts_ncells_nnodes(mk)[:2]

        write_s = lambda ss: vtuf.write(ss.encode('utf-8'))
        write_s('<Piece NumberOfPoints="{0}" NumberOfCells="{1}">\n'
                .format(npts, ncells))
        write_s('<Points>\n')

        # Write vtk DaraArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s('<DataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}" '
                    'format="appended" offset="{3}"/>\n'
                    .format(n, t, c, off))

            off += 4 + s

            # Write ends/starts of vtk file objects
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            elif i == 3:
                write_s('</Cells>\n<PointData>\n')

        # Write end of vtk element data
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_parallel_header(self, vtuf):
        names, types, comps = self._get_array_attrs()

        write_s = lambda ss: vtuf.write(ss.encode('utf-8'))
        write_s('<PPoints>\n')

        # Write vtk DaraArray headers
        for i, (n, t, s) in enumerate(zip(names, types, comps)):
            write_s('<PDataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}"/>\n'.format(n, t, s))

            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            elif i == 3:
                write_s('</PCells>\n<PPointData>\n')

        write_s('</PPointData>\n')

    def _write_data(self, vtuf, mk, soln):
        nsvpts = self.nsvpts[mk]
        nvars = soln.shape[1]
        neles = soln.shape[2]

        vtu_con, vtu_off, vtu_typ = self.vtu[mk]

        # Calculate solution at node locations of vtu elements
        vsol = np.dot(self.soln_vtu_op[mk], soln.reshape(-1, nvars*neles))
        vsol = vsol.reshape(nsvpts, nvars, -1).swapaxes(0, 1)

        # Write element node locations to file
        self._write_darray(self.vpts[mk].swapaxes(0, 1), vtuf, self.dtype)

        # Write vtu node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32)
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Primitive and visualisation variable maps
        vvars = OrderedDict(sorted(self.visvarmap.items(), key=lambda t: t[0]))

        # Convert from conservative to primitive variables
        vsol = np.array(self.elementscls.conv_to_pri(vsol, self.cfg))

        if self.export_gradients:
            mesh = self.mesh[mk]

            name = self.mesh_inf[mk][0]
            nspts, neles = mesh.shape[:2]

            # Dimensions
            ndims = self.ndims

            shapecls = subclass_where(BaseShape, name=name)

            # Shape
            soln_b = shapecls(nspts, self.cfg)

            ele = self.elementscls(shapecls, mesh, self.cfg)

            # Dimensions
            nvars = ele.nvars
            nupts = ele.nupts

            # smats, rjacs
            smats, djacs = ele._get_smats(soln_b.upts, True)
            rjacs = 1.0/djacs


            # tgard (ndim, nupts, nvars, neles)
            tgrad = np.dot(soln_b.opmat('M4'), soln.swapaxes(0, 1)
                           ).reshape(ndims, nupts, nvars, neles)

            # Eigensum
            grad = np.einsum('ijkl,kjml,jl->ijml', smats, tgrad, rjacs
                             ).reshape(ndims*nupts, -1)

            soln_vtu_op = self.soln_vtu_op[mk]
            # Interpolate gradient to nodes of vtu elements
            if self.ndims == 2:
                grad_vtu_op = block_diag((soln_vtu_op, soln_vtu_op))
            else:
                grad_vtu_op = block_diag((soln_vtu_op, soln_vtu_op,
                                          soln_vtu_op))

            vgrd = np.dot(grad_vtu_op, grad)

            # Rearrange the gradient matrix
            if self.ndims == 2:
                vgrd = np.concatenate((np.array_split(vgrd, 2)[0],
                                       np.array_split(vgrd, 2)[1]), axis=1)
            else:
                vgrd = np.concatenate((np.array_split(vgrd, 3)[0],
                                       np.array_split(vgrd, 3)[1],
                                       np.array_split(vgrd, 3)[2]), axis=1)

            vgrd = vgrd.reshape(nsvpts, nvars*self.ndims, -1
                                ).swapaxes(0, 1)
            # Concatenate solution and gradient arrays
            vsol = np.concatenate((vsol, vgrd), axis=0)

        # Write out the various fields
        for vnames in vvars.values():
            ix = [self.outvarmap.index(vn) for vn in vnames]
            self._write_darray(vsol[ix].T, vtuf, self.dtype)


class BaseShapeSubDiv(object):
    vtk_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)

    @classmethod
    def subcells(cls, n):
        pass

    @classmethod
    def subcelloffs(cls, n):
        return np.cumsum([cls.vtk_nodes[t] for t in cls.subcells(n)])

    @classmethod
    def subcelltypes(cls, n):
        return np.array([cls.vtk_types[t] for t in cls.subcells(n)])

    @classmethod
    def subnodes(cls, n):
        pass


class TensorProdShapeSubDiv(BaseShapeSubDiv):
    @classmethod
    def subnodes(cls, n):
        conbase = np.array([0, 1, n + 2, n + 1])

        # Extend quad mapping to hex mapping
        if cls.ndim == 3:
            conbase = np.hstack((conbase, conbase + (1 + n)**2))

        # Calculate offset of each subdivided element's nodes
        nodeoff = np.zeros((n,)*cls.ndim, dtype=np.int)
        for dim, off in enumerate(np.ix_(*(range(n),)*cls.ndim)):
            nodeoff += off*(n + 1)**dim

        # Tile standard element node ordering mapping, then apply offsets
        internal_con = np.tile(conbase, (n**cls.ndim, 1))
        internal_con += nodeoff.T.flatten()[:, None]

        return np.hstack(internal_con)


class QuadShapeSubDiv(TensorProdShapeSubDiv):
    name = 'quad'
    ndim = 2

    @classmethod
    def subcells(cls, n):
        return ['quad']*(n**2)


class HexShapeSubDiv(TensorProdShapeSubDiv):
    name = 'hex'
    ndim = 3

    @classmethod
    def subcells(cls, n):
        return ['hex']*(n**3)


class TriShapeSubDiv(BaseShapeSubDiv):
    name = 'tri'

    @classmethod
    def subcells(cls, n):
        return ['tri']*(n**2)

    @classmethod
    def subnodes(cls, n):
        conlst = []

        for row in range(n, 0, -1):
            # Lower and upper indices
            l = (n - row)*(n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extent list
            conlst.extend([subin, subex])

        return np.hstack(conlst)


class TetShapeSubDiv(BaseShapeSubDiv):
    name = 'tet'

    @classmethod
    def subcells(cls, nsubdiv):
        return ['tet']*(nsubdiv**3)

    @classmethod
    def subnodes(cls, nsubdiv):
        conlst = []
        jump = 0

        for n in range(nsubdiv, 0, -1):
            for row in range(n, 0, -1):
                # Lower and upper indices
                l = (n - row)*(n + row + 3) // 2 + jump
                u = l + row + 1

                # Lower and upper for one row up
                ln = (n + 1)*(n + 2) // 2 + l - n + row
                un = ln + row

                rowm1 = np.arange(row - 1)[..., None]

                # Base offsets
                offs = [(l, l + 1, u, ln), (l + 1, u, ln, ln + 1),
                        (u, u + 1, ln + 1, un), (u, ln, ln + 1, un),
                        (l + 1, u, u+1, ln + 1), (u + 1, ln + 1, un, un + 1)]

                # Current row
                conlst.extend(rowm1 + off for off in offs[:-1])
                conlst.append(rowm1[:-1] + offs[-1])
                conlst.append([ix + row - 1 for ix in offs[0]])

            jump += (n + 1)*(n + 2) // 2

        return np.hstack(np.ravel(c) for c in conlst)


class PriShapeSubDiv(BaseShapeSubDiv):
    name = 'pri'

    @classmethod
    def subcells(cls, n):
        return ['pri']*(n**3)

    @classmethod
    def subnodes(cls, n):
        # Triangle connectivity
        tcon = TriShapeSubDiv.subnodes(n).reshape(-1, 3)

        # Layer these rows of triangles to define prisms
        loff = (n + 1)*(n + 2) // 2
        lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in range(n)]

        return np.hstack(np.hstack(l).flat for l in lcon)


class PyrShapeSubDiv(BaseShapeSubDiv):
    name = 'pyr'

    @classmethod
    def subcells(cls, n):
        cells = []

        for i in range(n, 0, -1):
            cells += ['pyr']*(i**2 + (i - 1)**2)
            cells += ['tet']*(2*i*(i - 1))

        return cells

    @classmethod
    def subnodes(cls, nsubdiv):
        lcon = []

        # Quad connectivity
        qcon = [QuadShapeSubDiv.subnodes(n + 1).reshape(-1, 4)
                for n in range(nsubdiv)]

        # Simple functions
        def _row_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*i + j + 1)
                             for i in range(a, n + b)
                             for j in range(n - 1)])

        def _col_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*(i + 1) + j)
                             for i in range(n - 1)
                             for j in range(a, n + b)])

        u = 0
        for n in range(nsubdiv, 0, -1):
            l = u
            u += (n + 1)**2

            lower_quad = qcon[n - 1] + l
            upper_pts = np.arange(n**2) + u

            # First set of pyramids
            lcon.append([lower_quad, upper_pts])

            if n > 1:
                upper_quad = qcon[n - 2] + u
                lower_pts = np.hstack(range(k*(n + 1)+1, (k + 1)*n + k)
                                      for k in range(1, n)) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack(np.column_stack(l).flat for l in lcon)
