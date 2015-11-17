# -*- coding: utf-8 -*-

from collections import OrderedDict
import os
import shelve

import numpy as np

from pyfr.h5writer import H5Writer
from pyfr.inifile import Inifile
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin


class TStatsPlugin(BasePlugin):
    name = 'tstats'
    systems = ['*']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        basedir = self.cfg.getpath(cfgsect, 'basedir', '.')
        basename = self.cfg.get(cfgsect, 'basename', raw=True)
        self.params = OrderedDict(
            (k, v) for k, v in self.cfg.items(cfgsect, raw=True).items()
            if k.startswith('avg-')
        )

        # Copy the cfg, and add the field-names
        self.cfg = Inifile(self.cfg.tostr())
        self.cfg.set('soln-fieldnames', 'field-names',
                     ', '.join(self.params.keys()))

        self._writer = H5Writer(intg, basedir, basename, 'soln',
                                nvars=len(self.params))

        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.dt_bkp = self.cfg.getfloat(cfgsect, 'dt-bkp')

        self.nsteps = self.cfg.getfloat(cfgsect, 'nsteps')

        self.tbkp_next = intg.tcurr + self.dt_bkp
        self.nbkp = 0

        intg.call_plugin_dt(self.dt_out)
        plocs = intg.system.ele_ploc_upts

        self.prev_t = intg.tcurr
        self.prev_soln = [self._process(s, p, intg)
                          for s, p in zip(intg.soln, plocs)]

        order = self.cfg.getint('solver', 'order')
        prank = intg.rallocs.prank
        self.bkp_base = '__bkp/accm-{}-r{}-p{}'.format(
            intg.mesh_uuid, order, prank)
        if not os.path.exists('__bkp'):
            os.makedirs('__bkp')

        bkp0 = shelve.open('{}-{}'.format(self.bkp_base, 0))
        bkp1 = shelve.open('{}-{}'.format(self.bkp_base, 1))

        if 'prev_t' in bkp0 and abs(bkp0['prev_t']-self.prev_t) < intg.dtmin:
            self.avg_start_t = bkp0['avg_start_t']
            self.accm_soln = bkp0['accm_soln']

        elif 'prev_t' in bkp1 and abs(bkp1['prev_t']-self.prev_t) < intg.dtmin:
            self.avg_start_t = bkp1['avg_start_t']
            self.accm_soln = bkp1['accm_soln']

        else:
            self.avg_start_t = intg.tcurr
            self.accm_soln = [np.zeros_like(s) for s in self.prev_soln]

        bkp0.close()
        bkp1.close()

        self.tout_next = self.avg_start_t + self.dt_out

    def __call__(self, intg):
        time_to_write = abs(self.tout_next - intg.tcurr) < intg.dtmin
        time_to_bckup = abs(self.tbkp_next - intg.tcurr) < intg.dtmin

        # Accumulate every nsteps and every time the file needs to be written
        if (intg.nacptsteps % self.nsteps == 0) or time_to_write or \
                time_to_bckup:
            dt_prev = intg.tcurr - self.prev_t

            plocs = intg.system.ele_ploc_upts
            current_soln = [self._process(s, p, intg)
                            for s, p in zip(intg.soln, plocs)]

            if self.prev_soln is not None:
                for a, c, p in zip(self.accm_soln, current_soln,
                                   self.prev_soln):
                    a += (c + p) * dt_prev * 0.5

            self.prev_soln = current_soln
            self.prev_t = intg.tcurr

        # Check if it is time to write
        if time_to_write:
            for a in self.accm_soln:
                a *= 1.0/(intg.tcurr-self.avg_start_t)

            stats = Inifile()
            stats.set('tstats', 'start-time', self.avg_start_t)
            stats.set('tstats', 'end-time', intg.tcurr)

            metadata = dict(config=self.cfg.tostr(),
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)

            self._writer.write(self.accm_soln, metadata, intg.tcurr)

            # Reset for next accumulation
            self.avg_start_t = intg.tcurr
            self.accm_soln = [np.zeros_like(s) for s in self.prev_soln]

            self.tout_next += self.dt_out

        if time_to_write or time_to_bckup:
            bkup = shelve.open('{}-{}'.format(self.bkp_base, self.nbkp))
            bkup.clear()

            bkup['prev_t'] = self.prev_t
            bkup['avg_start_t'] = self.avg_start_t
            bkup['accm_soln'] = self.accm_soln

            bkup.close()

            self.nbkp ^= 1
            self.tbkp_next += self.dt_bkp

    def _process(self, soln, plocs, intg):
        # Constants
        local_vars = self.cfg.items_as('constants', float)

        # The primitives
        elecls = intg.system.elementscls
        local_vars.update(
            zip(elecls.privarmap[intg.system.ndims],
                elecls.conv_to_pri(soln.swapaxes(0, 1), self.cfg))
        )

        # The current time
        local_vars['t'] = intg.tcurr

        # The coordinates
        local_vars.update(dict(zip('xyz'[:self.ndims], plocs.swapaxes(0, 1))))

        # Evaluate
        params = [npeval(p, local_vars) for p in self.params.values()]

        return np.array(params).swapaxes(0, 1)
