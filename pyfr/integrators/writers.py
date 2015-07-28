# -*- coding: utf-8 -*-

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
        parallel = (h5py.get_config().mpi and
                    h5py.version.version_tuple >= (2, 5) and
                    not self.cfg.getbool('soln-output', 'serial-h5', False))

        if parallel:
            self._write = self._write_parallel

        else:
            self._write = self._write_serial
            if rank == root:
                self._mpi_rbufs = mpi_rbufs = []
                self._mpi_rreqs = mpi_rreqs = []

                for mrank, meleinfo in enumerate(eleinfo):
                    for tag, (etype, dims) in enumerate(meleinfo):
                        if mrank != root:
                            rbuf = np.empty(dims, dtype=self.backend.fpdtype)
                            rreq = comm.Recv_init(rbuf, mrank, tag)

                            mpi_rbufs.append(rbuf)
                            mpi_rreqs.append(rreq)

    def output(self, data):
        # Determine the output path
        path = self._get_output_path()

        # Delegate to _write to do the actual outputting
        self._write(path, data)

        # Increment the output number
        self.nout += 1

    def _get_output_path(self):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=self.tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _write_parallel(self, path, data):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(path, 'w', driver='mpio', comm=comm) as h5file:
            for group, solnmap, metadata in data:
                # Collect shapes of current rank
                shapes = [(k, v.shape) for k, v in solnmap.items()]

                all_shapes = comm.gather(shapes, root=root)
                if rank == root:
                    shape_dict = dict(s for p in all_shapes for s in p)
                else:
                    shape_dict = None

                # Dictionary of shapes from all ranks
                shape_dict = comm.bcast(shape_dict)
                smap = {}
                for name, shape in shape_dict.items():
                    smap[name] = h5file.create_dataset(
                        group + '/' + name, shape, dtype=self.backend.fpdtype
                    )

                comm.barrier()

                for name, sol in solnmap.items():
                    smap[name][:] = sol

                # Metadata information has to be transferred to all the ranks
                if rank == root:
                    mmap = [(k, len(v.encode()))
                            for k, v in metadata.items()]
                else:
                    mmap = None

                for name, size in comm.bcast(mmap, root=root):
                    d = h5file.create_dataset(name, (),
                                              dtype='S{}'.format(size))

                    if rank == root:
                        d.write_direct(np.array(metadata[name], dtype='S'))

                comm.barrier()

    def _write_serial(self, path, data):
        from mpi4py import MPI

        comm, rank, root = get_comm_rank_root()
        outdict = {}

        for group, solnmap, metadata in data:
            names = comm.gather(tuple(solnmap.keys()), root=root)
            if rank != root:
                for tag, buf in enumerate(solnmap.values()):
                    comm.Send(buf.copy(), root, tag)
            else:
                names = [group + '/' + i for p in names for i in p]

                # Recv all of the non-local solution mats
                MPI.Prequest.Startall(self._mpi_rreqs)
                MPI.Prequest.Waitall(self._mpi_rreqs)

                # Combine local and MPI data

                solns = it.chain(solnmap.values(), self._mpi_rbufs)

                # Convert any metadata to ASCII
                metadata = {k: np.array(v, dtype='S')
                            for k, v in metadata.items()}

                # Create the output dictionary
                outdict.update(dict(zip(names, solns), **metadata))

        with h5py.File(path, 'w') as h5file:
            for k, v in outdict.items():
                h5file[k] = v
