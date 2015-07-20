# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BasePlugin(object, metaclass=ABCMeta):
    name = None
    systems = None

    @abstractmethod
    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError('System {0} not supported by plugin {1}'
                               .format(intg.system.name, self.name))

    @abstractmethod
    def __call__(self, intg):
        pass


class CatalystPlugin(BasePlugin):
    name = 'catalyst'
    systems = ['*']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

    def __call__(self, intg):
        if intg.nsteps % self.nsteps == 0:
            if any(np.isnan(np.sum(s)) for s in intg.soln):
                raise RuntimeError('NaNs detected at t = {0}'
                                   .format(intg.tcurr))
