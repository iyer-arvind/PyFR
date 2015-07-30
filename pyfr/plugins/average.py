# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from pyfr.h5writer import H5Writer
from pyfr.plugins.base import BasePlugin


class AveragePlugin(BasePlugin, H5Writer):
    name = 'average'
    systems = ['*']

    def __init__(self, intg, cfgsect, *args, **kwargs):
        BasePlugin.__init__(self, intg, cfgsect)
        H5Writer.__init__(self, intg, cfgsect)

        self.writeevery = self.cfg.getint(self.cfgsect, 'write-every')

        self.averages = []
        self.solnprev = None
        self.dtprev = 0
        self.avglist = OrderedDict()
        self.callcount = 0
        self.prevwrite = None

    def handle(self, intg):
        # If this is not the first iteration, we can integrate
        if self.prevwrite is None:
            self.averages = [np.zeros_like(s) for s in intg.soln]
            self.prevwrite = intg.tcurr
            self.solnprev = intg.soln
        else:
            dt = intg.tcurr - self.tprev
            for avg, soln in zip(self.averages, self.solnprev):
                avg += 0.5*(self.dtprev + dt)*soln
            self.dtprev = dt
            self.solnprev = intg.soln
            if (self.callcount % self.writeevery) == 0:
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
                self.write(avgmap, metadata, intg.tcurr)
                self.averages = [np.zeros_like(s) for s in intg.soln]
                self.dtprev = 0
        self.callcount += 1
