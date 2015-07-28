# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.nputil import range_eval
from pyfr.util import proxylist


class MergeTimes(object):
    def __init__(self):
        self._nextvals = []

    def append(self, lst, handler):
        self._nextvals.append((lst.pop(0), lst, handler))

    def __iter__(self):
        return self

    def __next__(self):
        if not self._nextvals:
            raise StopIteration()

        t = self._nextvals[0][0]
        idx, objs = zip(*((idx, obj) for idx, obj in enumerate(self._nextvals)
                          if obj[0] == t))

        self._nextvals = [oi for ii, oi in enumerate(self._nextvals)
                          if ii not in idx]

        self._nextvals.extend((o[1].pop(0), o[1], o[2]) for o in objs if o[1])

        self._nextvals.sort(key=lambda x: x[0])

        return t, proxylist(o[2] for o in objs)


class BaseIntegrator(object, metaclass=ABCMeta):
    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        self.backend = backend
        self.rallocs = rallocs
        self.cfg = cfg

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 't0', 0.0)

        # Output times
        self.tout = sorted(range_eval(cfg.get('soln-output', 'times')))
        self.tend = self.tout[-1]

        # Current time; defaults to tstart unless resuming a simulation
        if initsoln is None or 'stats' not in initsoln:
            self.tcurr = self.tstart
        else:
            stats = Inifile(initsoln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')

            # Cull already written output times
            self.tout = [t for t in self.tout if t > self.tcurr]

        # Ensure no time steps are in the past
        if self.tout[0] < self.tcurr:
            raise ValueError('Output times must be in the future')

        self.tlist = MergeTimes()
        self.tlist.append(self.tout, self.write_solution)

        # Determine the amount of temp storage required by thus method
        nreg = self._stepper_nregs

        # Construct the relevant mesh partition
        self.system = systemcls(backend, rallocs, mesh, initsoln, nreg, cfg)

        # Extract the UUID of the mesh (to be saved with solutions)
        self._mesh_uuid = mesh['mesh_uuid']

        # Get a queue for subclasses to use
        self._queue = backend.queue()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        comm, rank, root = get_comm_rank_root()

        # Sum to get the global number over all partitions
        self._gndofs = comm.allreduce(ndofs, op=get_mpi('sum'))

    def _kernel(self, name, nargs):
        # Transpose from [nregs][neletypes] to [neletypes][nregs]
        transregs = zip(*self._regs)

        # Generate an kernel for each element type
        kerns = proxylist([])
        for tr in transregs:
            kerns.append(self.backend.kernel(name, *tr[:nargs]))

        return kerns

    def _prepare_reg_banks(self, *bidxes):
        for reg, ix in zip(self._regs, bidxes):
            reg.active = ix

    @abstractmethod
    def step(self, t, dt):
        pass

    @abstractmethod
    def advance_to(self, t):
        pass

    @abstractmethod
    def output(self, data):
        pass

    @abstractproperty
    def _controller_needs_errest(self):
        pass

    @abstractproperty
    def _stepper_has_errest(self):
        pass

    @abstractproperty
    def _stepper_nfevals(self):
        pass

    @abstractproperty
    def _stepper_nregs(self):
        pass

    @abstractproperty
    def _stepper_order(self):
        pass

    def run(self):
        for t, handlers in self.tlist:
            # Advance to time t
            solns = self.advance_to(t)

            self.output(handlers(self, solns))

    def write_solution(self, intg, solns):

        # Map solutions to elements types
        solnmap = OrderedDict(zip(
            ("soln_{}_p{}".format(e, self.rallocs.prank)
                for e in self.system.ele_types),
            solns))

        # Collect statistics
        stats = Inifile()
        self.collect_stats(stats)

        metadata = dict(config=self.cfg.tostr(),
                        stats=stats.tostr(),
                        mesh_uuid=self._mesh_uuid)
        # Output
        return '/', solnmap, metadata

    def collect_stats(self, stats):
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
