# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from pyfr.plugins.base import BasePlugin


class AveragePlugin(BasePlugin):
    name = 'average'
    systems = ['*']

    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')
        self.writeskip = self.cfg.getint(self.cfgsect, 'writeskip')

        self.tout = intg.tout[::self.writeskip]

        self.averages = []
        self.solnprev = None
        self.tprev = None
        self.dtprev = 0
        self.prevwrite = intg.tcurr
        self.avglist = OrderedDict()

    def __call__(self, intg):
        if not self.tout:
            return

        if ((intg.nacptsteps % self.nsteps == 0) or
                (intg.tcurr >= self.tout[0])):
            # If this is not the first iteration, we can integrate
            if self.tprev is None:
                self.averages = [np.zeros_like(s) for s in intg.soln]
            else:
                dt = intg.tcurr - self.tprev
                for avg, soln in zip(self.averages, self.solnprev):
                    avg += 0.5*(self.dtprev + dt)*soln

                self.dtprev = dt

            self.solnprev = intg.soln
            self.tprev = intg.tcurr

    def write(self, intg, solns):
        dt = intg.tcurr - self.tprev
        metadata = {}
        avgmap = {}
        for avg, soln in zip(self.averages, intg.soln):
            avg += 0.5*dt*soln
            avgnames = ('avg_{}_p{}'.format(e, intg.rallocs.prank)
                        for e in intg.system.ele_types)

            avgmap = OrderedDict(zip(avgnames, self.averages))
            metadata = {'tstart': str(self.prevwrite),
                        'tend': str(intg.tcurr)}

            self.prevwrite = intg.tcurr
            self.dtprev = 0

        self.averages = [np.zeros_like(s) for s in intg.soln]
        return 'averages', avgmap, metadata
