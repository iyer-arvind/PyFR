# -*- coding: utf-8 -*-

import itertools as it
import os

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root


class H5Writer(object):
    def __init__(self, intg, cfgsect, *args, **kwargs):
        cfg = intg.cfg

        # Base output directory and file name
        self._basedir = cfg.getpath(cfgsect, 'basedir', '.')
        self._basename = cfg.get(cfgsect, 'basename', raw=True)

        self.nout = 0

        system = intg.system

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the type and shape of each element in the partition
        etypes, shapes = system.ele_types, system.ele_shapes

        # Gather this information onto the root rank
        eleinfo = comm.gather(zip(etypes, shapes), root=root)

        # Deciding if parallel
        parallel = (h5py.get_config().mpi and
                    h5py.version.version_tuple >= (2, 5) and
                    not cfg.getbool(cfgsect, 'serial-h5', False))

        self.fpdtype = intg.backend.fpdtype

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
                            rbuf = np.empty(dims, dtype=self.fpdtype)
                            rreq = comm.Recv_init(rbuf, mrank, tag)

                            mpi_rbufs.append(rbuf)
                            mpi_rreqs.append(rreq)

    def write(self, datamap, metadata, tcurr):
        # Determine the output path
        path = self._get_output_path(tcurr)

        # Delegate to _write to do the actual outputting
        self._write(path, datamap, metadata)

        # Increment the output number
        self.nout += 1

    def _get_output_path(self, tcurr):
        # Substitute %(t) and %(n) for the current time and output number
        fname = self._basename % dict(t=tcurr, n=self.nout)

        # Append the '.pyfrs' extension
        if not fname.endswith('.pyfrs'):
            fname += '.pyfrs'

        return os.path.join(self._basedir, fname)

    def _write_parallel(self, path, datamap, metadata):
        print('parallel')
        comm, rank, root = get_comm_rank_root()
        # Collect shapes of current rank
        shapes = [(k, v.shape) for k, v in datamap.items()]
        all_shapes = comm.gather(shapes, root=root)

        if rank == root:
            shape_dict = dict(s for p in all_shapes for s in p)
        else:
            shape_dict = None

        # Dictionary of shapes from all ranks
        shape_dict = comm.bcast(shape_dict)

        with h5py.File(path, 'w', driver='mpio', comm=comm) as h5file:
            smap = {}
            for name, shape in shape_dict.items():
                smap[name] = h5file.create_dataset(
                    name, shape, dtype=self.fpdtype
                )

            for name, sol in datamap.items():
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

    def _write_serial(self, path, datamap, metadata):
        print('serial')
        from mpi4py import MPI

        comm, rank, root = get_comm_rank_root()
        outdict = {}

        names = comm.gather(tuple(datamap.keys()), root=root)

        if rank != root:
            for tag, buf in enumerate(datamap.values()):
                comm.Send(buf.copy(), root, tag)
        else:
            names = [i for p in names for i in p]

            # Recv all of the non-local solution mats
            MPI.Prequest.Startall(self._mpi_rreqs)
            MPI.Prequest.Waitall(self._mpi_rreqs)

            # Combine local and MPI data

            solns = it.chain(datamap.values(), self._mpi_rbufs)

            # Convert any metadata to ASCII
            metadata = {k: np.array(v, dtype='S')
                        for k, v in metadata.items()}

            # Create the output dictionary
            outdict.update(dict(zip(names, solns), **metadata))

        with h5py.File(path, 'w') as h5file:
            for k, v in outdict.items():
                h5file[k] = v
