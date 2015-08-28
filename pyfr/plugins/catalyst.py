# -*- coding: utf-8 -*-
import sys

import numpy as np
import paraview
from paraview import numpy_support
import paraview.servermanager as pvsm
paraview.options.batch = True
paraview.options.symmetric = True

import paraview.simple
import vtk
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


        try:
            import vtkPVServerManagerApplicationPython as ApplicationPython
        except Exception:
            paraview.print_error(
                "Error: Cannot import vtkPVServerManagerApplicationPython")

        if not CorePython.vtkProcessModule.GetProcessModule():
            pvoptions = None
            if paraview.options.batch:
                pvoptions = CorePython.vtkPVOptions()
                pvoptions.SetProcessType(CorePython.vtkPVOptions.PVBATCH)
                if paraview.options.symmetric:
                    pvoptions.SetSymmetricMPIMode(True)
            ApplicationPython.vtkInitializationHelper.Initialize(
                sys.executable,
                CorePython.vtkProcessModule.PROCESS_BATCH,
                pvoptions
            )

        if pvsm.vtkSMProxyManager.GetVersionMajor() != 4 or \
           pvsm.vtkSMProxyManager.GetVersionMinor() < 2:
            print 'Must use ParaView v4.2 or greater'
            sys.exit(0)

        self.coProcessor = catalyst.vtkCPProcessor()
        pm = paraview.servermanager.vtkProcessModule.GetProcessModule()


        self.ele_map = intg.system.ele_map

        points = vtk.vtkPoints()

        self.grid = vtk.vtkUnstructuredGrid()
        self.grid.Allocate(np.sum(s.neles for s in intg.system.eles))

        self.soln_vtu_op = []
        self.pts_off = [0]

        for s, e in zip(intg.system.eles, self.ele_map):
            shapecls = subclass_where(BaseShape, name=e)
            nvpts = shapecls.nspts_from_order(self.divisor+1)
            nspts, ndims, neles = s.nspts, s.ndims, s.neles
            
            vtu_b = shapecls(nvpts, self.cfg)

            mesh_vtu_op = s.basis.sbasis.nodal_basis_at(vtu_b.spts)

            pts = np.dot(mesh_vtu_op, s.eles.reshape(nspts, -1))
            pts = pts.reshape(nvpts, -1, ndims)

            if ndims == 2:
                pts = np.append(pts, np.zeros(pts.shape[:-1])[..., None],
                                axis=2)
            
            pts = pts.swapaxes(0, 1)
            for ei in range(neles):
                for p in range(nvpts):
                    points.InsertNextPoint(pts[ei, p, 0],
                                           pts[ei, p, 1],
                                           pts[ei, p, 2])

            self.pts_off.append(self.pts_off[-1]+nvpts*neles)
        self.grid.SetPoints(points)

        # for s, e, off in zip(intg.system.eles, self.ele_map, self.pts_off[:-1]):
        #     shapecls = subclass_where(BaseShape, name=e)
        #     nvpts = shapecls.nspts_from_order(self.divisor+1)
        #     subdvcls = subclass_where(pyfrpv.BaseShapeSubDiv, name=e)
        #     nspts, ndims, neles = s.nspts, s.ndims, s.neles
        #     nodes = subdvcls.subnodes(self.divisor)
        #     vtu_b = shapecls(nvpts, self.cfg)
        #     soln_vtu_op = s.basis.ubasis.nodal_basis_at(vtu_b.spts)
        #     vtu_con = np.tile(nodes, (neles, 1))
        #     vtu_con += (np.arange(neles)*nvpts)[:, None]
        #
        #     vtkt, nppc = pyfrpv.ParaviewWriter.vtk_types[e]
        #     vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)
        #     for t, cp in zip(vtu_typ, vtu_con):
        #         cp += off
        #         piece = np.split(cp, cp.shape[0]/nppc)
        #         for p in piece:
        #             self.grid.InsertNextCell(t, nppc, p)
        #             self.soln_vtu_op.append(soln_vtu_op)

        pipeline = pythoncatalyst.vtkCPPythonScriptPipeline()
        pipeline.Initialize(self.script)
        self.coProcessor.AddPipeline(pipeline)

    def __call__(self, intg):
        if intg.nsteps % self.nsteps != 0: 
            return
        print ("PLOTTING")
        co_processor = self.coProcessor
        data_description = catalyst.vtkCPDataDescription()
        data_description.SetTimeData(intg.tcurr, intg.nacptsteps)
        data_description.AddInput('input')
        # data = []
        # for soln, op, s in zip(intg.soln, self.soln_vtu_op, intg.system.eles):
        #     nspts, ndims, neles = s.nspts, s.ndims, s.neles
        #     sol = np.dot(op, soln.reshape(op.shape[1], -1))
        #     sol = sol.reshape([sol.shape[0], nspts, -1])
        #     pyfrpv.component_to_physical_soln(
        #         sol, self.cfg.getfloat('constants', 'gamma'))
        #
        #     data.append(np.array(sol[:, 1].T).reshape([-1]))
        # data = np.hstack(data)
        # print data.shape
        # pressure = paraview.numpy_support.numpy_to_vtk(data)
        # pressure.SetName("Pressure")
        #
        # self.grid.GetPointData().SetScalars(pressure)

        data_description.GetInputDescriptionByName("input").SetGrid(self.grid)
        co_processor.CoProcess(data_description)

    def finalize(self, intg):
        self.coProcessor.Finalize()
        ApplicationPython.vtkInitializationHelper.Finalize()
