# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict
import itertools as it
import os

import h5py
import numpy as np

from pyfr.integrators.base import BaseIntegrator
from pyfr.mpiutil import get_comm_rank_root


class H5Writer(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Base output directory and file name
        self._basedir = self.cfg.getpath('soln-output', 'basedir', '.')
        self._basename = self.cfg.get('soln-output', 'basename', raw=True)

        # Output counter (incremented each time output() is called)
        self.nout = 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes, shapes = self.system.ele_types, self.system.ele_shapes

        # Gather this information onto the root rank
        eleinfo = comm.gather(zip(etypes, shapes), root=root)

        # Deciding if parallel
        parallel = h5py.get_config().mpi
        parallel &= h5py.version.version_tuple[:2] >= (2, 5)
        parallel &= not self.cfg.getbool('soln-output', 'serial-h5', False)

        if parallel:
            self._write = self._write_parallel

            if rank == root:
                sollist = []
                sizelist = defaultdict(
                    lambda: np.zeros([comm.Get_size(), 3], dtype='int32'))
                for mrank, meleinfo in enumerate(eleinfo):
                    prank = self.rallocs.mprankmap[mrank]
                    for etype, dims in meleinfo:
                        sollist.append(
                            (self._get_name_for_soln(etype, prank), dims))
                        sizelist[etype][prank, :] = dims

                offsetlist = OrderedDict()
                for e in sizelist:
                    offsetlist[e] =\
                        (tuple(sizelist[e][0, :2]),
                         np.hstack([[0], np.cumsum(sizelist[e][:, 2])]))
            else:
                sollist = None
                offsetlist = None
            self.sollist, self.offsetlist =\
                comm.bcast((sollist, offsetlist), root=root)

        else:
            self._write = self._write_serial

            if rank == root:
                self._mpi_rbufs = mpi_rbufs = []
                self._mpi_rreqs = mpi_rreqs = []
                self._mpi_names = mpi_names = []
                self._loc_names = loc_names = []

                for mrank, meleinfo in enumerate(eleinfo):
                    prank = self.rallocs.mprankmap[mrank]
                    for tag, (etype, dims) in enumerate(meleinfo):
                        name = self._get_name_for_soln(etype, prank)

                        if mrank == root:
                            loc_names.append(name)
                        else:
                            rbuf = np.empty(dims, dtype=self.backend.fpdtype)
                            rreq = comm.Recv_init(rbuf, mrank, tag)

                            mpi_rbufs.append(rbuf)
                            mpi_rreqs.append(rreq)
                            mpi_names.append(name)

    def output(self, solnmap, stats):
        comm, rank, root = get_comm_rank_root()

        # Convert the config and stats objects to strings
        if rank == root:
            metadata = dict(config=self.cfg.tostr(),
                            stats=stats.tostr(),
                            mesh_uuid=self._mesh_uuid)
        else:
            metadata = None

        # Determine the output path
        path = self._get_output_path()

        # Delegate to _write to do the actual outputting
        self._write(path, solnmap, metadata)

        # Increment the output number
        self.nout += 1

    def _get_output_path(self):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=self.tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _get_name_for_soln(self, etype, prank=None):
        prank = self.rallocs.prank if prank is None else prank
        return 'soln_{}_p{}'.format(etype, prank)

    def _write_parallel(self, path, solnmap, metadata):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as h5file:
            smap = {}
            if self.cfg.getbool('soln-output', 'combined-arrays', False):
                for e in self.offsetlist:
                    smap[e] = h5file.create_dataset(
                        'soln_{}'.format(e), self.offsetlist[e][0] +
                        (self.offsetlist[e][1][-1],),
                        dtype=self.backend.fpdtype)
                    for prank in range(comm.Get_size()):
                        offs = self.offsetlist[e][1][prank]
                        offe = self.offsetlist[e][1][prank+1]
                        if(offe > offs):
                            name = self._get_name_for_soln(e, prank)
                            smap[e].attrs[name] = np.array([offs, offe])

                for e in solnmap:
                    data = np.array(solnmap[e])
                    offs = self.offsetlist[e][1][self.rallocs.prank]
                    offe = self.offsetlist[e][1][self.rallocs.prank+1]
                    shape = self.offsetlist[e][0] + (offe-offs, )
                    start = tuple(0 for s in shape[:-1]) + (offs, )
                    mspace = h5py.h5s.create_simple(shape)
                    fspace = smap[e].id.get_space()
                    fspace.select_hyperslab(start, shape)
                    smap[e]._id.write(mspace, fspace, data)

            else:
                for s, shape in self.sollist:
                    smap[s] = h5file.create_dataset(
                        s, shape, dtype=self.backend.fpdtype
                    )

                for e, sol in solnmap.items():
                    s = self._get_name_for_soln(e, self.rallocs.prank)
                    smap[s][:] = sol

            # Metadata information has to be transferred to all the ranks
            if rank == root:
                mmap = [(k, len(v.encode()))
                        for k, v in metadata.items()]
            else:
                mmap = None

            mmap = comm.bcast(mmap, root=root)
            for name, size in mmap:
                d = h5file.create_dataset(name, (), dtype='S{}'.format(size))
                if rank == root:
                    d.write_direct(np.array(metadata[name]).astype('S'))

    def _write_serial(self, path, solnmap, metadata):
        from mpi4py import MPI

        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for tag, buf in enumerate(solnmap.values()):
                comm.Send(buf.copy(), root, tag)
        else:
            # Recv all of the non-local solution mats
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)

            # Combine local and MPI data
            names = it.chain(self._loc_names, self._mpi_names)
            solns = it.chain(solnmap.values(), self._mpi_rbufs)

            # Create the output dictionary
            outdict = dict(zip(names, solns), **metadata)

            with h5py.File(path, 'w') as h5file:
                for k, v in outdict.items():
                    h5file.create_dataset(k, data=v)
