# -*- coding: utf-8 -*-

import numpy as np


from pyfr.plugins.spatial_average import SpatialAverage
from pyfr.plugins.base import init_csv
from pyfr.mpiutil import get_comm_rank_root


class PIDController(object):
    def __init__(self, intg, cfgsect):
        cfg = intg.cfg
        self.p_constant = cfg.getfloat(cfgsect, 'Kp')
        self.i_constant = cfg.getfloat(cfgsect, 'Ki')
        self.d_constant = cfg.getfloat(cfgsect, 'Kd')

        self.err_accm = cfg.getfloat(cfgsect, 'err_accm', 0)
        self.err_prev = cfg.getfloat(cfgsect, 'err_prev', 0)

        comm, rank, root = get_comm_rank_root()

        if rank == root and cfg.get(cfgsect, 'file', ''):
            self.outf = init_csv(cfg, cfgsect, 'n,t,err,err_accm,p,i,d,corr')

        else:
            self.outf = None

        self.n = 0
        self.t_prev = intg.tcurr

    def __call__(self, err, tcurr):
        dt = tcurr - self.t_prev

        p = err * self.p_constant
        if self.n == 0:
            d = 0
            self.err_accm = err*0.5*dt

        else:
            d = (err-self.err_prev) / dt * self.d_constant
            self.err_accm += (err + self.err_prev) * dt * 0.5

        i = self.err_accm * self.i_constant

        corr = -(p + i + d)

        if self.outf:
            s = (self.n, tcurr, err, self.err_accm, p, i, d, corr)
            print(','.join(str(c) for c in s), file=self.outf)
            self.outf.flush()
        
        self.n += 1
        self.err_prev = err
        self.t_prev = tcurr

        return corr


class BulkFlowController(SpatialAverage):
    name = 'bulk_flow_controller'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        intg.cfg.set(cfgsect, 'avg-u', 'u')
        intg.cfg.set(cfgsect, 'dt-out', '0')
        intg.cfg.set(cfgsect, 'basename', '.')
        super().__init__(intg, cfgsect, suffix)

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 1)
        self.ubulk = self.cfg.getfloat(cfgsect, 'ubulk')

        self.pid = PIDController(intg, cfgsect)
        self.err_avg = 0.0

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps:
            return

        self.eles_scal_upts_inb.active = intg._idxcurr
        u_bulk = self._run(intg.tcurr, False)
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            err = u_bulk-self.ubulk

            self.err_avg = self.err_avg*0.1 + abs(err)*0.9
            alpha = (1+np.tanh(1.0*(np.log(abs(self.err_avg))+18)))/2

            if self.cfg.hasopt(self.cfgsect, 'corr'):
                corr_prev = self.cfg.getfloat(self.cfgsect, 'corr')
            else:
                corr_prev = 0.00

            corr = alpha*self.pid(err, intg.tcurr) + (1-alpha)*corr_prev

            print(' ', self.err_avg, self.nsteps)

        else:
            corr = None

        self.cfg.set(self.cfgsect, 'corr', comm.bcast(corr, root=root))
