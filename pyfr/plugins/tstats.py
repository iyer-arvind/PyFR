# -*- coding: utf-8 -*-

from collections import OrderedDict
from threading import Thread

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
        self.nsteps = self.cfg.getfloat(cfgsect, 'nsteps')

        self.tout_next = intg.tcurr + self.dt_out

        intg.call_plugin_dt(self.dt_out)

        self.avg_start_t = intg.tcurr
        self.prev_t = intg.tcurr

        self.mesh_uuid = intg.mesh_uuid
        self.elecls = intg.system.elementscls
        self.plocs = intg.system.ele_ploc_upts
        self.ndims = intg.system.ndims

        self.prev_soln = [self._process(s, p, intg.tcurr)
                          for s, p in zip(intg.soln, self.plocs)]

        self.accm_soln = [np.zeros_like(s) for s in self.prev_soln]

        self.thread = None

    def __call__(self, intg):
        time_to_write = abs(self.tout_next - intg.tcurr) < intg.dtmin

        # Accumulate every nsteps and every time the file needs to be written
        if (intg.nacptsteps % self.nsteps == 0) or time_to_write:
            soln = [a.copy() for a in intg.soln]
            tcurr = intg.tcurr

            # Always wait for previous thread to finish
            if self.thread:
                self.thread.join()

            if time_to_write:
                self.run(soln, tcurr)

                for a in self.accm_soln:
                    a *= 1.0/(tcurr-self.avg_start_t)

                stats = Inifile()
                stats.set('tstats', 'start-time', self.avg_start_t)
                stats.set('tstats', 'end-time', tcurr)

                metadata = dict(config=self.cfg.tostr(),
                                stats=stats.tostr(),
                                mesh_uuid=self.mesh_uuid)

                self._writer.write(self.accm_soln, metadata, tcurr)

                # Reset for next accumulation
                self.avg_start_t = tcurr
                self.accm_soln = [np.zeros_like(s) for s in self.prev_soln]

                self.tout_next += self.dt_out

            else:
                self.thread = Thread(target=self.run,
                                     args=(soln, tcurr))
                self.thread.start()

    def run(self, soln, tcurr):
        dt_prev = tcurr - self.prev_t

        current_soln = [self._process(s, p, tcurr)
                        for s, p in zip(soln, self.plocs)]

        if self.prev_soln is not None:
            for a, c, p in zip(self.accm_soln, current_soln,
                               self.prev_soln):
                a += (c + p) * dt_prev * 0.5

        self.prev_soln = current_soln
        self.prev_t = tcurr

    def _process(self, soln, plocs, tcurr):
        # Constants
        local_vars = self.cfg.items_as('constants', float)

        # The primitives

        local_vars.update(
            zip(self.elecls.privarmap[self.ndims],
                self.elecls.conv_to_pri(soln.swapaxes(0, 1), self.cfg))
        )

        # The current time
        local_vars['t'] = tcurr

        # The coordinates
        local_vars.update(dict(zip('xyz'[:self.ndims],
                                   plocs.swapaxes(0, 1))))

        # Evaluate
        params = [npeval(p, local_vars) for p in self.params.values()]

        return np.array(params).swapaxes(0, 1)
