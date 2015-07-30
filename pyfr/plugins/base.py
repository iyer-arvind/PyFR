# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BasePlugin(object, metaclass=ABCMeta):
    name = None
    systems = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # No output by default
        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps', -1)
        self.nsteps = self.nsteps if self.nsteps > 0 else None

        self.dt = self.cfg.getfloat(self.cfgsect, 'dt', -1)
        self.dt = self.dt if self.dt > 0 else None

        if self.dt is not None:
            self.tprev = intg.tcurr - self.dt

        assert self.nsteps is not None or self.dt is not None,\
            'both dt and nsteps cannot be none'

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError('System {0} not supported by plugin {1}'
                               .format(intg.system.name, self.name))

    def __call__(self, intg):
        if ((self.nsteps is not None and
                intg.nacptsteps % self.nsteps == 0) or
            (self.dt is not None and
                abs(self.tprev + self.dt - intg.tcurr) < intg.dtmin)):
            self.handle(intg)
            self.tprev = intg.tcurr

    @abstractmethod
    def handle(self, intg):
        pass
