# -*- coding: utf-8 -*-

from pyfr.plugins.average import AveragePlugin
from pyfr.plugins.base import BasePlugin
from pyfr.plugins.fluidforce import FluidForcePlugin
from pyfr.plugins.nancheck import NaNCheckPlugin
from pyfr.plugins.residual import ResidualPlugin
from pyfr.plugins.sampler import SamplerPlugin
from pyfr.plugins.solutionwriter import SolutionWriterPlugin

from pyfr.util import subclass_where


def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
