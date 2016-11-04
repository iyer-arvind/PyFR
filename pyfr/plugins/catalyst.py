# -*- coding: utf-8 -*-

from collections import OrderedDict
from ctypes import *
import importlib.util
import time

from mpi4py import MPI
import numpy as np
from pycuda import compiler

from pyfr.backends.base import ComputeKernel
from pyfr.backends.cuda.provider import  get_grid_for_block
from pyfr.plugins.base import BasePlugin
from pyfr.ctypesutil import load_library
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclass_where
import os
import configparser


# Contains relevant data pertaining to all instances of a single cell type
class MeshDataForCellType(Structure):
    _fields_ = [
        ('nVerticesPerCell', c_int),
        ('nCells', c_int),

        ('vertices', c_void_p),

        ('nSubdividedCells', c_int),

        ('con', c_void_p),
        ('off', c_void_p),
        ('type', c_void_p)
    ]

class SolutionDataForCellType(Structure):
    _fields_ = [
        ('ldim', c_int),
        ('lsdim', c_int),
        ('soln', c_void_p)
    ]

class CatalystData(Structure):
    _fields_ = [
        ('nCellTypes', c_int),
        ('meshData', POINTER(MeshDataForCellType)),
        ('solutionData', POINTER(SolutionDataForCellType)),
        ('isovalues', POINTER(c_float)),
        ('niso', c_uint),
        ('metadata', c_bool),
        ('eye', POINTER(c_float)),
        ('ref', POINTER(c_float)),
        ('vup', POINTER(c_float)),
    ]

        
class Camera(object):
    def __init__(self, spec_file, offset, scale):
        data = np.loadtxt(spec_file, delimiter=' ')
        if len(data.shape) == 1:
            data = data[np.newaxis,...]
        print(data.shape)

        self.time = np.hstack(([0], np.cumsum(data[:,0])))[:-1]
        self.eye = data[:,1:4]
        self.vup = data[:,4:7]
        self.ref = data[:,7:10]

        self.scale = scale
        self.offset = offset

    def __call__(self, t):
        tt = (t-self.offset)*self.scale
        I = np.where(tt<self.time)[0].tolist()
        if tt < 0:
            return self.eye[0,:], self.ref[0,:], self.vup[0,:]
        if not I:
            return self.eye[-1,:], self.ref[-1,:], self.vup[-1,:]

        I = I[0]
        f = (tt - self.time[I-1])/(self.time[I]-self.time[I-1])

        return (self.eye[I-1] + f*(self.eye[I]-self.eye[I-1]),
                self.ref[I-1] + f*(self.ref[I]-self.ref[I-1]),
                self.vup[I-1] + f*(self.vup[I]-self.vup[I-1]))


class CatalystPlugin(BasePlugin):
    name = 'catalyst'
    systems = ['euler', 'navier-stokes']

    def __init__(self, intg, *args, **kwargs):
        _start = time.time()
    
        super().__init__(intg, *args, **kwargs)

        self.divisor = self.cfg.getint(self.cfgsect, 'divisor', 3)
        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

        outputfile = self.cfg.get(self.cfgsect, 'outputfile')
        c_outputfile = create_string_buffer(bytes(outputfile, encoding='utf_8'))

        hostname = self.cfg.get(self.cfgsect, 'hostname')
        c_hostname = create_string_buffer(bytes(hostname, encoding='utf_8'))

        port = self.cfg.getint(self.cfgsect, 'port');


        self.image_dir= self.cfg.get(self.cfgsect, 'image-dir', '.')

        # 'metadata_out' indicates the user wants to output per-TS metadata.
        try:
            self.metadata = self.cfg.getbool(self.cfgsect, 'metadata_out')
        except configparser.NoOptionError:
            self.metadata = False

        # parse out the [optional] eye/ref/vup parameters.
        def literal_or_vec3f(key):
            # We use NaN to mean "value was not present."
            a = [ float('NaN'), float('NaN'), float('NaN') ]
            try:
                a = self.cfg.getliteral(self.cfgsect, key)
                if len(a) != 3:
                    raise Exception(key + " must be a 3-element list")
            except configparser.NoOptionError:
                pass
            return a

        if self.cfg.get(self.cfgsect, 'camera-spec', ''):
            cameras = [i.strip() 
                       for i in self.cfg.get(self.cfgsect, 'camera-spec').split(',')]

            offset = self.cfg.getfloat(self.cfgsect, 'camera-t-off')
            scale = self.cfg.getfloat(self.cfgsect, 'camera-t-scale')
            
            self.camera = OrderedDict((camera.rsplit('.', 1)[0],Camera(camera, offset, scale)) for camera in cameras)

            eye, ref, vup = (-10,0,0), (0,0,0), (0,1,0)

        else:
            self.camera = None
            eye = literal_or_vec3f('eye')
            ref = literal_or_vec3f('ref')
            vup = literal_or_vec3f('vup')

        self.eye = (c_float * 3)()
        self.ref = (c_float * 3)()
        self.vup = (c_float * 3)()
        self.eye[0] = eye[0]; self.eye[1] = eye[1]; self.eye[2] = eye[2]
        self.ref[0] = ref[0]; self.ref[1] = ref[1]; self.ref[2] = ref[2]
        self.vup[0] = vup[0]; self.vup[1] = vup[1]; self.vup[2] = vup[2]



        # Load catalyst library
        self.catalyst = load_library('pyfr_catalyst_fp32')
        print(self.catalyst)

        self.backend = backend = intg.backend
        self.mesh = intg.system.mesh

        # Amount of subdivision to perform
#        comm = MPI.COMM_WORLD
#        self.divisor = comm.Get_size()

        # Allocate a queue on the backend
        self._queue = backend.queue()

        # Solution arrays
        self.eles_scal_upts_inb = inb = intg.system.eles_scal_upts_inb

        # Check Dobule precision
        prec = self.cfg.get('backend', 'precision', 'double')
        if prec == 'double':
            # Change precision as single
            backend.fpdtype = np.dtype('single')

            # Converter from dp to sp
            kern = compiler.SourceModule("""
                __global__ void cvt(const int nrow, const int nvars,
                const int neles, const double *src, const int lsdsrc,
                float *dst, const int lsddst)
                {
                  int i = blockIdx.x*blockDim.x + threadIdx.x;
                  if (i < neles) {
                  for (int j=0; j < nrow; j++) {
                  for (int k=0; k < nvars; k++) {
                  dst[(nvars*j+k)*lsddst + i] = (float)src[(nvars*j+k)*lsdsrc + i];
                  }
                  }
                  }
                }
                """).get_function('cvt')
            kern.prepare('iiiPiPi')

            def gen_cvtkern(soln, ssoln):
                nrow, nvars, neles = soln.ioshape
                block = (192, 1, 1)
                grid = get_grid_for_block(block, soln.ncol)

                class CVTKern(ComputeKernel):
                    def run(self, queue):
                        kern.prepared_async_call(grid, block,
                                                 queue.cuda_stream_comp,
                                                 nrow, nvars, neles,
                                                 soln, soln.leadsubdim,
                                                 ssoln, ssoln.leadsubdim)

                return CVTKern()

        # CVT kerns
        ckerns = []

        # Prepare the mesh data and solution data
        meshData, solnData, kerns = [], [], []
        for etype, solnmat in zip(intg.system.ele_types, inb):
            p, solnop = self._prepare_vtu(etype, intg.rallocs.prank)

            # Allocate on the backend
            vismat = backend.matrix((p.nVerticesPerCell, self.nvars, p.nCells),
                                    tags={'align'})

            solnop = backend.const_matrix(solnop)
            backend.commit()

            # Populate the soln field and dimension info
            s = SolutionDataForCellType(ldim = vismat.leaddim,
                                        lsdim = vismat.leadsubdim,
                                        soln = vismat.data)

            if prec == 'double':
                # Prepare to convert dp to sp
                ssolnmat = backend.matrix(solnmat.ioshape, tags={'align'})
                ckerns.append(gen_cvtkern(solnmat, ssolnmat))

                # Prepare the matrix multiplication kernel
                k = backend.kernel('mul', solnop, ssolnmat, out=vismat)
            else:
                # Prepare the matrix multiplication kernel
                k = backend.kernel('mul', solnop, solnmat, out=vismat)

            # Append
            meshData.append(p)
            solnData.append(s)
            kerns.append(k)

        # 'isovalues' in the config file should be a list.
        iv = self.cfg.getliteral(self.cfgsect, 'isovalues',[1.0])
        niso = len(iv)
        isovalues = (c_float * niso)()
        for i in range(len(isovalues)):
            isovalues[i] = iv[i]

        
        # Save the pieces
        catalystData = []
        catalystData.append(
            CatalystData(nCellTypes = len(meshData),
             meshData = (MeshDataForCellType*len(meshData))(*meshData),
             solutionData = (SolutionDataForCellType*len(solnData))(*solnData),
             isovalues = isovalues,
             niso = niso,
             metadata = c_bool(self.metadata),
             eye=self.eye, ref=self.ref, vup=self.vup)
        )
        self._catalystData = (CatalystData*len(catalystData))(*catalystData)

        # Wrap the kernels in a proxy list
        self._interpolate_upts = proxylist(kerns)
        self._conver_sp = proxylist(ckerns)


        # Finally, initialize Catalyst
        print('Initializing catalyst plugin.... ')
        self._data = self.catalyst.CatalystInitialize(c_hostname,
                                                      c_int(int(port)),
                                                      c_outputfile,
                                                      self._catalystData)

        fields = {'rho':0, 'u':1, 'v':2, 'w':3, 'e':1, 'Q':9, 'vel_mag':5, 'grad_rho':6, 'grad_v':7, 'grade': 8}

        cntb = fields[self.cfg.get(self.cfgsect, 'contour-by', 'Q')]
        self.catalyst.CatalystSetFieldToContourBy(cntb)

        clb = fields[self.cfg.get(self.cfgsect, 'contour-color-by', 'rho')]
        self.catalyst.CatalystSetFieldToColorBy(clb, 1)

        clb = fields[self.cfg.get(self.cfgsect, 'slice-color-by', 'rho')]
        self.catalyst.CatalystSetFieldToColorBy(clb, 2)

        origin = (c_float*3)()
        normal = (c_float*3)()

        o = self.cfg.getliteral(self.cfgsect, 'slice-origin', [0, 0, 0])
        n = self.cfg.getliteral(self.cfgsect, 'slice-normal', [0, 0, 1])
        for i in range(3):
            origin[i] = o[i]
            normal[i] = n[i]

        self.catalyst.CatalystSetSlicePlanes(origin, normal, 1, 0)

            

        # Image Resolution
        img_res = (c_uint32 * 2)()
        img_res[0], img_res[1] = self.cfg.getliteral(self.cfgsect, 'image-size', '400, 600')
        self.catalyst.CatalystImageResolution(self._catalystData, img_res)
            
        color = (c_float *3)()
        color[0], color[1], color[2] = self.cfg.getliteral(self.cfgsect, 'image-bgcolor', '1.0, 1.0, 1.0')
        self.catalyst.CatalystBGColor(self._data, color)

        col_min, col_max = self.cfg.getliteral(self.cfgsect, 'contour-color-range', '0.01, 0.99')
        self.catalyst.CatalystSetColorRange(self._data, c_double(col_min), c_double(col_max), 1)

        col_min, col_max = self.cfg.getliteral(self.cfgsect, 'slice-color-range', '0.01, 0.99')
        self.catalyst.CatalystSetColorRange(self._data, c_double(col_min), c_double(col_max), 2)

        color_map = self.cfg.getliteral(self.cfgsect, 'contour-color-map',[(0.1, 255, 255, 255, 255), (0.9, 0, 0, 0, 255)])
        print(color_map)

        n_cols = len(color_map)
        colors = (c_uint8*(n_cols*4))()
        pivots = (c_float*n_cols)()
        for i, (p, r, g, b, a) in enumerate(color_map):
            print(p, r, g, b, a)
            pivots[i] = p
            colors[i*4+0], colors[i*4+1], colors[i*4+2], colors[i*4+3] = r, g, b, a
        
        print('Range: ', col_min, col_max)
        print('Colors: ', [colors[i] for i in range(4*n_cols)])
        print('Pivots: ', [pivots[i] for i in range(n_cols)])
        self.catalyst.CatalystSetColorTable(self._data, colors, pivots, c_size_t(n_cols), 1)


        color_map = self.cfg.getliteral(self.cfgsect, 'slice-color-map',[(0.1, 255, 255, 255, 255), (0.9, 0, 0, 0, 255)])
        print(color_map)

        n_cols = len(color_map)
        colors = (c_uint8*(n_cols*4))()
        pivots = (c_float*n_cols)()
        for i, (p, r, g, b, a) in enumerate(color_map):
            print(p, r, g, b, a)
            pivots[i] = p
            colors[i*4+0], colors[i*4+1], colors[i*4+2], colors[i*4+3] = r, g, b, a
        
        print('Range: ', col_min, col_max)
        print('Colors: ', [colors[i] for i in range(4*n_cols)])
        print('Pivots: ', [pivots[i] for i in range(n_cols)])
        self.catalyst.CatalystSetColorTable(self._data, colors, pivots, c_size_t(n_cols), 2)


        cp = (self.cfg.getliteral(self.cfgsect, 'contour-clip-plane-1', [(0, 0.8, 0), (-0.8660, 0.5, 0)]),
              self.cfg.getliteral(self.cfgsect, 'contour-clip-plane-2', [(0,-0.2, 0), ( 0.8660,-0.5, 0)]))


        # plane / origin-normal / coordinate
        cpc = [(c_float*3)(), (c_float*3)()],[(c_float*3)(), (c_float*3)()]

        for p in range(2):
            for i in range(2):
                for j in range(3):
                    cpc[p][i][j] = cp[p][i][j]
        
        print('Clip planes: ', cp)
        def pc(a):
            return '[{},{},{}]'.format(a[0],a[1],a[2]) 

        print(pc(cpc[0][0]), pc(cpc[0][1]), pc(cpc[1][0]), pc(cpc[1][1]))
        self.catalyst.CatalystSetClipPlanes(cpc[0][0], cpc[0][1], cpc[1][0], cpc[1][1])


        if prec == 'double':
            # Roll back to double precision
            self.backend.fpdtype = np.dtype('double')

        print('Catalyst plugin initialization time: {}s'.format(time.time()-_start))


    def _prepare_vtu(self, etype, part):
        from pyfr.writers.vtk import BaseShapeSubDiv

        mesh = self.mesh['spt_{0}_p{1}'.format(etype, part)]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)
        subdvcls = subclass_where(BaseShapeSubDiv, name=etype)

        # Dimensions
        # tjc: nspts: number of points in the element type
        # tjc: neles: number of elements of this type
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.divisor)
        nsvpts = len(svpts)

        # Shape
        soln_b = shapecls(nspts, self.cfg)

        # Generate the operator matrices
        mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
        soln_vtu_op = soln_b.ubasis.nodal_basis_at(svpts)

        # Calculate node locations of vtu elements
        vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Reorder and cast
        vpts = vpts.swapaxes(0, 1).astype(np.float32, order='C')

        # Perform the sub division
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare vtu cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]
        vtu_con = vtu_con.astype(np.int32, order='C')

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]
        vtu_off = vtu_off.astype(np.int32, order='C')

        # Tile vtu cell type numbers
        vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)
        vtu_typ = vtu_typ.astype(np.uint8, order='C')

        # Construct the meshDataForCellType
        meshDataForCellType = \
        MeshDataForCellType(nVerticesPerCell=nsvpts,
                            nCells=neles,
                            vertices=vpts.ctypes.data_as(c_void_p),
                            nSubdividedCells=len(vtu_typ),
                            con=vtu_con.ctypes.data,
                            off=vtu_off.ctypes.data,
                            type=vtu_typ.ctypes.data)

        # Retain the underlying NumPy objects
        meshDataForCellType._vpts = vpts
        meshDataForCellType._vtu_con = vtu_con
        meshDataForCellType._vtu_off = vtu_off
        meshDataForCellType._vtu_typ = vtu_typ

        return meshDataForCellType, soln_vtu_op

    def __call__(self, intg):
        _start = time.time()
        if np.isclose(intg.tcurr, intg.tend):

            # Configure the input bank
            self.eles_scal_upts_inb.active = intg._idxcurr

            # Configure the input bank
            self.eles_scal_upts_inb.active = intg._idxcurr

            # Convert dp to sp
            if self._conver_sp:
                self._queue % self._conver_sp()

            # Interpolate to the vis points
            self._queue % self._interpolate_upts()

            self.catalyst.CatalystCoProcess(c_double(intg.tcurr),intg.nacptsteps,self._data,c_bool(True))
            self.catalyst.CatalystFinalize(self._data)
            return

        if intg.nacptsteps % self.nsteps:
            return

        # Configure the input bank
        self.eles_scal_upts_inb.active = intg._idxcurr

        # Convert dp to sp
        if self._conver_sp:
            self._queue % self._conver_sp()

        # Interpolate to the vis points
        self._queue % self._interpolate_upts()

        if self.camera:
            for name, camera in self.camera.items():
                eye, ref, vup = camera(intg.tcurr)

                print(intg.tcurr, eye, ref, vup)

                self.eye[0] = eye[0]; self.eye[1] = eye[1]; self.eye[2] = eye[2]
                self.ref[0] = ref[0]; self.ref[1] = ref[1]; self.ref[2] = ref[2]
                self.vup[0] = vup[0]; self.vup[1] = vup[1]; self.vup[2] = vup[2]


                prefix = '{}/{}.'.format(self.image_dir, name)
                c_fnp = create_string_buffer(bytes(prefix, encoding='utf_8'))
                self.catalyst.CatalystFilenamePrefix(self._data, c_fnp)
                
                if 'contour' in name:
                    self.catalyst.CatalystSetViewToCoprocess(self._data, c_int(0))

                else:
                    self.catalyst.CatalystSetViewToCoprocess(self._data, c_int(1))

                
                self.catalyst.CatalystCamera(self._data, self.eye, self.ref, self.vup)

                self.catalyst.CatalystCoProcess(c_double(intg.tcurr), intg.nacptsteps,
                                                self._data, c_bool(False))


        else:

            self.catalyst.CatalystCoProcess(c_double(intg.tcurr), intg.nacptsteps,
                                            self._data, c_bool(False))


        if self.metadata:
            print('Catalyst plugin __call__ time: {}s'.format(time.time()-_start))
