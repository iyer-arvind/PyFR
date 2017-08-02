# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import init_csv
from pyfr.plugins.fluidforce import FluidForcePlugin

class PressureController(FluidForcePlugin):
    name = 'pressurecontroller'
    systems = ['euler', 'navier-stokes']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        
        self.c_outf = init_csv(self.cfg, cfgsect,
                               't, p, p_target, error, corr_i, corr_p, corr',
                               filekey='controller-file')
        
        # Viscous correction
        self.face_area = self.cfg.getfloat(cfgsect, 'area')
        self.target_pressure = self.cfg.getfloat(cfgsect, 'target-p')
        self.i_factor = self.cfg.getfloat(cfgsect, 'i-factor')
        self.p_factor = self.cfg.getfloat(cfgsect, 'p-factor')

        self.prevt = intg.tcurr
        self.prevd = None

        self.intg_d = 0
        self.cfg.p_corr = 0


    def __call__(self, intg):
        row = super().__call__(intg)
        if row is None:
            return

        comm, rank, root = get_comm_rank_root() 

        # Get the correction on the root rank
        if rank == root:
            dt = intg.tcurr - self.prevt

            # calculate the pressure (x-normal face)
            p = row[1]/self.face_area
            
            # the error
            d = self.target_pressure - p

            # correction is zero first time
            if self.prevd is None:
                corr = 0

            else:
                # get the integrated value
                self.intg_d += 0.5 * (d + self.prevd) * dt
                
                # calculate the correction
                corr_i = self.intg_d*self.i_factor
                corr_p = d * self.p_factor
                corr = corr_i + corr_p
             
                # print the stats
                print(intg.tcurr, p, self.target_pressure, d,
                      corr_i, corr_p, corr,
                      file=self.c_outf)
                
                # flush the file
                self.c_outf.flush()

            # save the time and error
            self.prevt = intg.tcurr
            self.prevd = d

        else:
            corr = None 
        
        # transfer the correction to all ranks
        self.cfg.p_corr = comm.bcast(corr, root=root)

