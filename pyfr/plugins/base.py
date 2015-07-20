# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BasePlugin(object):
    __metaclass__ = ABCMeta
    name = None
    systems = None

    @abstractmethod
    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg._system.ndims
        self.nvars = intg._system.nvars

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError('System {0} not supported by plugin {1}'
                               .format(intg.system.name, self.name))

    @abstractmethod
    def __call__(self, intg):
        pass

    def finalize(self, intg):
        pass
