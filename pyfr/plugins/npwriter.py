# -*- coding: utf-8 -*-

import os
import time
import pickle

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root


class NPWriterPlugin(BasePlugin):
    name = 'npwriter'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Construct the solution writer
        self.basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        self.basename = self.cfg.get(cfgsect, 'basename')
        
        # Output time step and next output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_next = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self(intg)
        else:
            self.tout_next += self.dt_out

    def __call__(self, intg):
        if abs(self.tout_next - intg.tcurr) > self.tol:
            return

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        metadata = dict(config=self.cfg.tostr(),
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        comm, rank, root = get_comm_rank_root()

        dirname = (self.basedir + '/' +
                   self.basename.format(t=intg.tcurr) + '.dir')

        if rank == root:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(dirname + '/metadata','wb') as fh:
                pickle.dump(metadata, fh)
    
        # Make sure that the directory is created   
        comm.Barrier()
        while not os.path.exists(dirname):
            time.sleep(1)

        for k, v in zip(intg.system.ele_types, intg.soln):
            print('Writing {}/soln_{}_p{}'.format(dirname, k, rank))
            np.save('{}/soln_{}_p{}'.format(dirname, k, rank), v)

        self.tout_next = intg.tcurr + self.dt_out
