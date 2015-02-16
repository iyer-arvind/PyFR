# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BasePartitioner(object):
    # Approximate element weighting table
    _ele_wts = {'quad': 3, 'tri': 2, 'tet': 2, 'hex': 6, 'pri': 4, 'pyr': 3}

    def __init__(self, partwts, opts={}):
        self.partwts = partwts

        # Parse the options list
        self.opts = {}
        for k, v in dict(self.dflt_opts, **opts).items():
            if k in self.int_opts:
                self.opts[k] = int(v)
            elif k in self.enum_opts:
                self.opts[k] = self.enum_opts[k][v]
            else:
                raise ValueError('Invalid partitioner option')

    #def _combine_mesh_parts(self, mesh):
    #    # Get the per-partition element counts
    #    pinf = mesh.partition_info

    #    # Shape points and element number offsets
    #    spts = defaultdict(list)
    #    offs = defaultdict(dict)

    #    for en, pn in pinf.items():
    #        for i, n in enumerate(pn):
    #            if n > 0:
    #                offs[en][i] = sum(s.shape[1] for s in spts[en])
    #                spts[en].append(mesh['spt_{0}_p{1}'.format(en, i)])

    #    def offset_con(con, pr):
    #        con = con.copy()

    #        for en, pn in pinf.items():
    #            if pn[pr] > 0:
    #                con['f1'][np.where(con['f0'] == en)] += offs[en][pr]

    #        return con

    #    # Connectivity
    #    intcon, mpicon, bccon = [], {}, defaultdict(list)

    #    for f in mesh:
    #        mi = re.match(r'con_p(\d+)$', f)
    #        mm = re.match(r'con_p(\d+)p(\d+)$', f)
    #        bc = re.match(r'bcon_(.+?)_p(\d+)$', f)

    #        if mi:
    #            intcon.append(offset_con(mesh[f], int(mi.group(1))))
    #        elif mm:
    #            l, r = int(mm.group(1)), int(mm.group(2))
    #            lcon = offset_con(mesh[f], l)

    #            if (r, l) in mpicon:
    #                rcon = mpicon.pop((r, l))
    #                intcon.append(np.vstack([lcon, rcon]))
    #            else:
    #                mpicon[l, r] = lcon
    #        elif bc:
    #            name, l = bc.group(1), int(bc.group(2))
    #            bccon[name].append(offset_con(mesh[f], l))

    #    # Concatenate these arrays to from the new mesh
    #    newmesh = {'con_p0': np.hstack(intcon)}

    #    for k, v in spts.items():
    #        newmesh['spt_{0}_p0'.format(k)] = np.hstack(v)

    #    for k, v in bccon.items():
    #        newmesh['bcon_{0}_p0'.format(k)] = np.hstack(v)

    #    return newmesh

    #def _combine_soln_parts(self, soln):
    #    newsoln = defaultdict(list)

    #    for f, (en, shape) in soln.array_info.items():
    #        newsoln['soln_{0}_p0'.format(en)].append(soln[f])

    #    newsoln = {k: np.dstack(v) for k, v in newsoln.items()}
    #    newsoln['config'] = soln['config']
    #    newsoln['stats'] = soln['stats']

    #    return newsoln

    def _construct_graph(self, mesh):
        # Edges of the dual graph
        con = mesh.getInterface('internal')
        con = np.hstack([con, con[::-1]])
        # Sort by the left hand side
        idx = np.lexsort([con['type'][0], con['ele'][0]])
        con = con[:,idx]

        # Left and right hand side element types/indicies
        lhs, rhs = con[['type', 'ele']]

        # Compute vertex offsets
        vtab = np.where(lhs[1:] != lhs[:-1])[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the element type/index to vertex number map
        vetimap = [tuple(lhs[i]) for i in vtab[:-1]]
        etivmap = {k: v for v, k in enumerate(vetimap)}

        # Prepare the list of edges for each vertex
        etab = np.array([etivmap[tuple(r)] for r in rhs])

        # Prepare the list of vertex and edge weights
        vwts = np.array([self._ele_wts[t] for t, i in vetimap])
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), vetimap

    def _partition_graph(self, graph, partwts):
        pass

    def _partition_spts(self, mesh, vparts, vetimap):
        # Get the shape point arrays from the mesh
        spt_p0 = {}
        for s in mesh.shapes():
            spt_p0[s] = mesh.getShapePoints(s)

        # Partition the shape points
        spt_px = defaultdict(list)

        for (etype, eidxg), part in zip(vetimap, vparts):
            spt_px[etype, part].append(spt_p0[etype][:,eidxg,:])

        # Stack
        return {'{0}-{1}'.format(*k): np.array(v).swapaxes(0, 1) for k, v in spt_px.items()}

    def _partition_map(self,vparts, vetimap):

        pMap=defaultdict(list)
        for (etype, eidxg), part in zip(vetimap, vparts):
            pMap["{0}-{1}".format(etype,part)].append(eidxg)

        return pMap

    #def _partition_soln(self, soln, vparts, vetimap):
    #    # Get the solution arrays from the file
    #    soln_p0 = {}
    #    for f in soln:
    #        if f.startswith('soln'):
    #            soln_p0[f.split('_')[1]] = soln[f]

    #    # Partition the solutions
    #    soln_px = defaultdict(list)
    #    for (etype, eidxg), part in zip(vetimap, vparts):
    #        soln_px[etype, part].append(soln_p0[etype][...,eidxg])

    #    # Stack
    #    return {'soln_{0}_p{1}'.format(*k): np.dstack(v)
    #            for k, v in soln_px.items()}

    def _partition_con(self, mesh, vparts, vetimap):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        bcon_px = defaultdict(list)

        # Global-to-local element index map
        eleglmap = defaultdict(list)
        pcounter = Counter()

        for (etype, eidxg), part in zip(vetimap, vparts):
            eleglmap[etype].append((part, pcounter[etype, part]))
            pcounter[etype, part] += 1

        # Generate the face connectivity
        con = mesh.getInterface('internal')
        for l, r in zip(*con):
            letype, leidxg, lfidx, lflags = l
            retype, reidxg, rfidx, rflags = r

            lpart, leidxl = eleglmap[letype][leidxg]
            rpart, reidxl = eleglmap[retype][reidxg]

            conl = (letype, leidxl, lfidx, lflags)
            conr = (retype, reidxl, rfidx, rflags)

            if lpart == rpart:
                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[lpart, rpart].append(conl)
                con_pxpy[rpart, lpart].append(conr)

        # Generate boundary conditions
        bcons=[b for b in mesh.getInterfaces() if b != "internal"]
        for b in bcons:
            for lpetype, leidxg, lfidx, lflags in mesh.getInterface(b):
                lpart, leidxl = eleglmap[lpetype][leidxg]
                conl = (lpetype, leidxl, lfidx, lflags)

                bcon_px[b, lpart].append(conl)

        # Output
        ret = {}

        for k, v in con_px.items():
            ret["internal-{0}".format(k)] = np.array(v, dtype=[('type', 'S4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')]).T

        for k, v in con_pxpy.items():
            ret["conn-{0}-{1}".format(k[0],k[1])] = np.array(v, dtype=[('type', 'S4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')])

        for k, v in bcon_px.items():
            ret["{0}-{1}".format(k[0],k[1])] =np.array(v, dtype=[('type', 'S4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')])

        return ret

    def partition(self, mesh):
        # Perform the partitioning
        if len(self.partwts) > 1:
            # Obtain the dual graph for this mesh
            graph, vetimap = self._construct_graph(mesh)

            # Partition the graph
            vparts = self._partition_graph(graph, self.partwts)

            # Partition the connectivity portion of the mesh
            interfaces = self._partition_con(mesh, vparts, vetimap)

            # Handle the shape points
            spts = self._partition_spts(mesh, vparts, vetimap)

            # Get the partition map
            pMap = self._partition_map(vparts, vetimap)
        # Short circuit
        else:
            raise NotImplementedError
            newmesh = mesh
        print (pMap)
        return interfaces,spts,pMap
