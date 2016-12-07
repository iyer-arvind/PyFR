#!/usr/bin/env python

from collections import defaultdict
import h5py
import numpy as np
import re
import uuid


def join_soln(d, po, eo, in_mesh, in_soln, out_mesh, out_soln):
    poff, eoff = po, eo
    print(poff, eoff)
    for k in in_mesh.keys():
        print(k)
        m = re.match(r'^bcon_([a-zA-Z0-9]+)_p([0-9])$', k)
        if m:
            bname, bpn = m.groups()
            bcond = np.array(in_mesh[k]).astype('U4,i4,i1,i1')
            bcond[:]['f1'] += np.array(list(eo[e] for e in bcond[:]['f0']), dtype='i4')
            out_mesh['bcon_{}_p{}'.format(bname, int(bpn)+po)] = bcond.astype('S4,i4,i1,i1')
            continue

        m = re.match(r'^con_p([0-9])$', k)
        if m:
            cp, = m.groups()
            con = np.array(in_mesh[k]).astype('U4,i4,i1,i1')
            con[:,0]['f1'] += np.array(list(eo[e] for e in con[:,0]['f0']), dtype='i4')
            con[:,1]['f1'] += np.array(list(eo[e] for e in con[:,1]['f0']), dtype='i4')
            out_mesh['con_p{}'.format(int(cp)+po)] = con.astype('S4,i4,i1,i1')
            continue

        m = re.match(r'^con_p([0-9])p([0-9])$', k)
        if m:
            cp1,cp2 = m.groups()
            con = np.array(in_mesh[k]).astype('U4,i4,i1,i1')
            con[:]['f1'] += np.array(list(eo[e] for e in con[:]['f0']), dtype='i4')
            out_mesh['con_p{}p{}'.format(int(cp1)+po, int(cp2)+po)] = con.astype('S4,i4,i1,i1')
            continue

        m = re.match(r'^spt_([a-zA-Z0-9]+)_p([0-9])$', k)
        if m:
            ename, epn = m.groups()
            spt = np.array(in_mesh[k])
            nupts, neles, ndims = spt.shape
            
            spt += d[np.newaxis, np.newaxis, :]
            print(d)
            out_mesh['spt_{}_p{}'.format(ename, int(epn)+po)] = spt

            out_soln['soln_{}_p{}'.format(ename, int(epn)+po)] = in_soln['soln_{}_p{}'.format(ename, int(epn))].value

            poff += 1
            #eoff[ename] += neles

            print(neles)
            continue

        if k == 'mesh_uuid':
            if po == 0:
                u = uuid.uuid4()
                out_mesh['mesh_uuid'] = out_soln['mesh_uuid'] = str(u)
                out_soln['config'] = in_soln['config'].value
                out_soln['stats'] = in_soln['stats'].value
            continue

        print(k)


    return poff, eoff


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--in-mesh',  type=str, required=True, help='input mesh')
    parser.add_argument('--count',  type=int, required=True, help='number of instances')
    parser.add_argument('--soln-name-template', type=str, required=True, help='solution name templated with "N"')
    parser.add_argument('--soln-offset', type=int, required=True, help='solution index offset')
    parser.add_argument('--out-mesh', type=str, required=True, help='output mesh')
    parser.add_argument('--out-soln', type=str, required=True, help='output solution')

    args = parser.parse_args()

    with h5py.File(args.in_mesh, 'r') as in_mesh, \
        h5py.File(args.out_mesh, 'w') as out_mesh, \
        h5py.File(args.out_soln, 'w') as out_soln:

        po = 0
        eo = defaultdict(lambda :0)
        for i in range(args.count):
            soln_name = args.soln_name_template.format(i+args.soln_offset)
            with h5py.File(soln_name, 'r') as soln:
                po, eo = join_soln(np.array([0, i*3, 0]), po, eo, in_mesh, soln, out_mesh, out_soln)

            out_soln.flush()
            out_mesh.flush()
