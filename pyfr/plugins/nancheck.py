# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin


class NaNCheckPlugin(BasePlugin):
    name = 'nancheck'
    systems = ['*']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            if any(np.isnan(np.sum(s)) for s in intg.soln):
                raise RuntimeError('NaNs detected at t = {0}'
                                   .format(intg.tcurr))
