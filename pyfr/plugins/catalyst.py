# -*- coding: utf-8 -*-

import sys

import numpy as np
import paraview
from paraview import numpy_support
import paraview.servermanager as pvsm
import paraview.simple
import vtk
import vtkParallelCorePython
import vtkPVCatalystPython as catalyst
import vtkPVClientServerCoreCorePython as CorePython
import vtkPVPythonCatalystPython as pythoncatalyst
import vtkPVServerManagerApplicationPython as ApplicationPython

from pyfr.plugins.base import BasePlugin
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers import paraview as pyfrpv

class CatalystPlugin(BasePlugin):
    name = 'catalyst'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        BasePlugin.__init__(self, intg, cfgsect, suffix)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')
        self.script = self.cfg.get(self.cfgsect, 'script')
        self.divisor = self.cfg.getint(self.cfgsect, 'divisor', 2)


        paraview.options.batch = True
        paraview.options.symmetric = True

        try:
            import vtkPVServerManagerApplicationPython as ApplicationPython
        except:
            paraview.print_error("Error: Cannot import vtkPVServerManagerApplicationPython")

        if not CorePython.vtkProcessModule.GetProcessModule():
            pvoptions = None
            if paraview.options.batch:
                pvoptions = CorePython.vtkPVOptions();
                pvoptions.SetProcessType(CorePython.vtkPVOptions.PVBATCH)
                if paraview.options.symmetric:
                    pvoptions.SetSymmetricMPIMode(True)
            ApplicationPython.vtkInitializationHelper.Initialize(sys.executable, CorePython.vtkProcessModule.PROCESS_BATCH, pvoptions)

        if pvsm.vtkSMProxyManager.GetVersionMajor() != 4 or \
           pvsm.vtkSMProxyManager.GetVersionMinor() < 2:
            print 'Must use ParaView v4.2 or greater'
            sys.exit(0)

        self.coProcessor = catalyst.vtkCPProcessor()
        pm = paraview.servermanager.vtkProcessModule.GetProcessModule()
        from mpi4py import MPI


        self.ele_map = intg._system.ele_map

        points = vtk.vtkPoints()

        self.grid = vtk.vtkUnstructuredGrid()
        self.grid.Allocate(np.sum(s.neles for s in intg._system.eles))

        self.soln_vtu_op = []
        self.pts_off = [0]

        for s, e in zip(intg._system.eles, self.ele_map):
            shapecls = subclass_where(BaseShape, name=e)
            subdvcls = subclass_where(pyfrpv.BaseShapeSubDiv, name=e)
            nvpts = shapecls.nspts_from_order(self.divisor+1)
            nspts, ndims, neles = s.nspts, s.ndims, s.neles
            
            vtu_b = shapecls(nvpts, self.cfg)

            mesh_vtu_op = s._basis.sbasis.nodal_basis_at(vtu_b.spts)
            soln_vtu_op = s._basis.ubasis.nodal_basis_at(vtu_b.spts)

            cells = subdvcls.subcells(self.divisor)
            nodes = subdvcls.subnodes(self.divisor)

            pts = np.dot(mesh_vtu_op, s.eles.reshape(nspts, -1))
            pts = pts.reshape(nvpts, -1, ndims)

            if ndims == 2:
                pts = np.append(pts, np.zeros(pts.shape[:-1])[..., None], axis=2)
            
            pts = pts.swapaxes(0, 1)
            vtu_con = np.tile(nodes, (neles, 1))
            vtu_con += (np.arange(neles)*nvpts)[:, None]

            for e in range(neles):
                for p in range(nvpts):
                    points.InsertNextPoint(pts[e,p,0], pts[e,p,1], pts[e,p,2])

            self.pts_off.append(self.pts_off[-1]+nvpts*neles)
        self.grid.SetPoints(points)
        for s, e, off in zip(intg._system.eles, self.ele_map, self.pts_off[:-1]):
            vtkt, nppc = pyfrpv.ParaviewWriter.vtk_types[e]
            vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)
            for t, cp in zip(vtu_typ,vtu_con):
                cp += off
                piece = np.split(cp,cp.shape[0]/nppc)
                for p in piece:
                    self.grid.InsertNextCell(t, nppc, p)

            self.soln_vtu_op.append(soln_vtu_op)


        pipeline = pythoncatalyst.vtkCPPythonScriptPipeline()
        pipeline.Initialize(self.script)
        self.coProcessor.AddPipeline(pipeline)



    def __call__(self, intg):
        if intg.nsteps % self.nsteps != 0: 
            return
        print ("PLOTTING")
        coProcessor = self.coProcessor
        dataDescription = catalyst.vtkCPDataDescription()
        dataDescription.SetTimeData(intg.tcurr, intg.nacptsteps)
        dataDescription.AddInput("input")
        for soln, op, s in zip(intg.soln, self.soln_vtu_op, intg._system.eles):
            nspts, ndims, neles = s.nspts, s.ndims, s.neles
            sol = np.dot(op, soln.reshape(op.shape[1], -1))
            sol = sol.reshape([sol.shape[0],nspts,-1])
            pyfrpv._component_to_physical_soln(sol, 
                    self.cfg.getfloat('constants', 'gamma'))
            data=np.array(sol[:,1].T).reshape([-1])

        pressure = paraview.numpy_support.numpy_to_vtk(data)
        pressure.SetName("Pressure")

        self.grid.GetPointData().SetScalars(pressure)

        dataDescription.GetInputDescriptionByName("input").SetGrid(self.grid)
        coProcessor.CoProcess(dataDescription)

    def finalize(self, intg):
        self.coProcessor.Finalize()
        ApplicationPython.vtkInitializationHelper.Finalize()
