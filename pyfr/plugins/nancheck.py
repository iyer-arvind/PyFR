# -*- coding: utf-8 -*-

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin
from pyfr.writers.native import NativeWriter

class NaNCheckPlugin(BasePlugin):
    name = 'nancheck'
    systems = ['*']

    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')
        
        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Construct the solution writer
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.')
        basename = self.cfg.get(self.cfgsect, 'basename', None)
        if basename:
            self._writer = NativeWriter(intg, self.nvars, basedir, basename,
                                        prefix='soln')
        else:
            self.writer = None

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            nan = np.zeros(1)
            nan[0] = 1 if any(np.isnan(np.sum(s)) for s in intg.soln) else 0

            comm, rank, root = get_comm_rank_root()
            comm.Allreduce(get_mpi('in_place'), nan, op=get_mpi('sum'))


            if nan[0] > 0:
                if self._writer:
                    stats = Inifile()
                    stats.set('data', 'fields', ','.join(self.fields))
                    stats.set('data', 'prefix', 'soln')
                    intg.collect_stats(stats)

                    # Prepare the metadata
                    metadata = dict(intg.cfgmeta,
                                    stats=stats.tostr(),
                                    mesh_uuid=intg.mesh_uuid)

                    # Write out the file
                    solnfname = self._writer.write(
                        intg.soln, metadata, intg.tcurr)

                raise RuntimeError('NaNs detected at t = {0}'
                                   .format(intg.tcurr))
