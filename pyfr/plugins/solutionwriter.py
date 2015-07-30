# -*- coding: utf-8 -*-

from collections import OrderedDict

from pyfr.h5writer import H5Writer
from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin


class SolutionWriterPlugin(BasePlugin, H5Writer):
    name = 'solutionwriter'
    systems = ['*']

    def __init__(self, intg, cfgsect, *args, **kwargs):
        BasePlugin.__init__(self, intg, cfgsect)
        H5Writer.__init__(self, intg, cfgsect)

    def handle(self, intg):
        solnnames = ('soln_{}_p{}'.format(e, intg.rallocs.prank)
                     for e in intg.system.ele_types)

        solnmap = OrderedDict(zip(solnnames, intg.soln))

        stats = Inifile()
        intg.collect_stats(stats)

        metadata = dict(config=self.cfg.tostr(),
                        stats=stats.tostr(),
                        mesh_uuid=intg._mesh_uuid)

        self.write(solnmap, metadata, intg.tcurr)
