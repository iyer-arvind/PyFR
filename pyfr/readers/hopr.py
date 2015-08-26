# -*- coding: utf-8 -*-

import numpy as np
import h5py

from pyfr.readers import BaseReader
from pyfr.shapes import BaseShape
from pyfr.util import subclasses


class HOPRReader(BaseReader):
    # Supported file types and extensions
    name = 'hopr'
    extn = ['.h5']

    def __init__(self, msh):
        super().__init__()
        if isinstance(msh, str):
            self.hpr = h5py.File(msh, 'r')
        else:
            self.hpr = h5py.File(msh.name, 'r')
        self.ng = self.hpr.attrs['Ngeo']

    def __get_ele_type(self, hopr_typ):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}
        lin, n_cpts = int(hopr_typ / 10), hopr_typ % 10

        if lin < 10:
            typ = {3: 'tri', 4: 'quad'}[n_cpts]
            spts = {0: n_cpts, 1: n_cpts,
                    2: basismap[typ].nspts_from_order(self.ng+1)}[lin]
        else:
            typ = {4: 'tet', 5: 'pyr', 6: 'pri', 8: 'hex'}[n_cpts]
            spts = {10: n_cpts, 11: n_cpts,
                    20: basismap[typ].nspts_from_order(self.ng+1)}[lin]

        return typ, spts

    def _to_raw_pyfrm(self):
        mesh = {}  # Holds the mesh
        t_map = {}  # The HOPR type to PyFR shape class map

        # Map of HOPR global element index to PyFR internal face index
        ge_map = {}

        get_map = {}  # Map of global element index to HOPR element type index

        print('Reading data..')
        # Read the data arrays to numpy arrays
        xyz = np.array(self.hpr['NodeCoords'])
        sides = np.array(self.hpr['SideInfo'])
        hpr_eles = np.array(self.hpr['ElemInfo'])

        print('Generating maps..')
        # Create the shape point arrays
        for htyp, count in self.hpr['ElemCounter']:
            if count > 0:
                typ, nspts = self.__get_ele_type(htyp)
                t_map[htyp] = typ
                mesh['spt_{0}_p0'.format(typ)] = np.zeros([nspts, count, 3])
                gidx = np.where(hpr_eles[:, 0] == htyp)[0]
                ge_map[typ] = {hei: i for i, hei in enumerate(gidx)}
                get_map.update({hei: typ for hei in gidx})

        bc_name = {i+1: 'bcon_{}_p0'.format(name.decode().strip())
                   for i, name in enumerate(self.hpr['BCNames'])}

        # Get the global side to internal and bc index map
        gf_map = {}
        # Create the connectivity arrays
        con_typ = [('f0', 'S4'), ('f1', 'i4'), ('f2', 'i1'), ('f3', 'i1')]
        for i in range(self.hpr['BCNames'].shape[0] + 1):
            bc_index = np.where(sides[:, 4] == i)[0]
            gf_map[i] = {gi-1: ii for ii, gi in
                         enumerate(sorted(set(abs(sides[bc_index, 1]))))}
            if i:
                name = bc_name[i]
                shp = [len(gf_map[i])]
            else:
                name = 'con_p0'
                shp = [2, len(gf_map[i])]

            mesh[name] = np.zeros(shp, dtype=con_typ)
        con = mesh['con_p0']

        print('Converting..')
        periodic = {}
        # Conversion loop
        for e, (htyp, _, side_b, side_e, node_b, node_e) in enumerate(hpr_eles):
            typ = t_map[htyp]
            eidx = ge_map[typ][e]

            mesh['spt_{0}_p0'.format(typ)][:, eidx, :] = xyz[node_b:node_e, :]

            for i in range(side_b, side_e):
                letyp = typ  # Left element type
                leidx = eidx  # Left element index
                lfli = i - side_b  # Left face local index

                shtyp, sgid, enbid, nblocf, bcid = sides[i]

                if bcid == 0:  # Internal interface
                    if sgid < 0:
                        continue  # Can write the reverse info as well

                    ifi = gf_map[0][sgid - 1]  # Internal face index

                    assert con[0, ifi]['f0'] == b''
                    assert con[1, ifi]['f0'] == b''

                    retyp = get_map[enbid-1]
                    reidx = ge_map[retyp][enbid-1]
                    rfli = int(nblocf/10) - 1

                    con[0, ifi] = (letyp, leidx, lfli, 0)
                    con[1, ifi] = (retyp, reidx, rfli, 0)

                elif nblocf == 0:  # External boundary
                    bc = bc_name[bcid]  # Name of the boundary
                    fi = gf_map[bcid][sgid - 1]  # Index of the face
                    mesh[bc][fi] = (letyp, leidx, lfli, 0)

                else:  # Periodic boundary
                    if sgid < 0:
                        continue

                    lbc = bc_name[bcid]

                    rfli = int(nblocf/10) - 1
                    if lbc not in periodic:
                        rbc = bc_name[sides[hpr_eles[enbid-1, 2]+rfli, 4]]
                        print(lbc, '->', rbc)
                        periodic[lbc] = 0
                        periodic[rbc] = 1
                        msh = np.vstack((mesh[lbc], mesh[rbc]))
                        mesh[lbc] = mesh[rbc] = msh

                    ifi = gf_map[bcid][sgid - 1]
                    assert mesh[lbc][0, ifi]['f0'] == b''
                    assert mesh[lbc][1, ifi]['f0'] == b''

                    retyp = get_map[enbid-1]
                    reidx = ge_map[retyp][enbid-1]
                    mesh[lbc][0, ifi] = (letyp, leidx, lfli, 0)
                    mesh[lbc][1, ifi] = (retyp, reidx, rfli, 0)

        if periodic:
            con = [con]
            con.extend([mesh[p] for p, d in periodic.items() if d == 0])
            for p in periodic:
                mesh.pop(p)
            mesh['con_p0'] = np.hstack(con)

        return mesh
