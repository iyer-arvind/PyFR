
from pyfr.mpiutil import get_comm_rank_root
import h5py
import sys
import numpy as np


from .pyfrio import BaseIO as IO


#######################################################################################
########### PARTITIONING CLASS
class H5Partitioning(IO.Partitioning):
    def __init__(self,name,io):
        self.group=io._file["mesh"]["partitionings"][name]
        super(self.__class__,self).__init__(name,io)
    
    def nPartitions(self):
        return len(self.group.keys())

    def getCompositeShapePoints(self,shp):
        shpl=[]
        for p in range(self.nPartitions()):
            part=self.getPartition(p)
            shpl.append(part.getShapePoints(shp))
        return np.concatenate(shpl,axis=1)

    #######################################################################################
    ########### PARTITION CLASS
    class Partition(IO.Partitioning.Partition):
        def __init__(self,partitioning,number):
            self.comm,self.rank,self.root=get_comm_rank_root()
            super(H5Partitioning.Partition,self).__init__(partitioning,number)
            self.group=self.partitioning.group["partition-%d"%number]

        # ====================================================================================
        # IO for shapes
        def getShapes(self):
            """Return the names of the shapes"""
            return self.group["shape-points"].keys()

        def getShapeCounts(self):
            """Return a dict of the shapes and the count of the elements of that shape"""
            shp=self.group["shape-points"]
            return {s:shp[s].shape[1] for s in shp}

        def getShapePoints(self,shape):
            return np.array(self.group["shape-points"][shape])


        # ====================================================================================
        # IO for internal interfaces, boundaries and connectivity
        def getConnectivities(self):
            return tuple(map(lambda x:int(x.replace('conn-',"")),filter(lambda x:x.startswith("conn-"),self.group["interfaces"].keys())))
        
        def getConnectivity(self,to):
            return self.__getInterface("conn-%d"%to)

        def getBoundaries(self):
            return list(map(lambda x:x.replace("bcon-",""),filter(lambda x:x.startswith("bcon-"),self.group["interfaces"].keys())))

        def getBoundary(self,bc):
            return self.__getInterface("bcon-%s"%bc)

        def getInternalInterface(self):
            return self.__getInterface("internal")

        def __getInterface(self,key):
            T=np.array(self.group['interfaces'][key])
            dtype=[('type', 'U4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')]
            U=np.zeros_like(T,dtype=dtype)
            for e,t in  dtype:
                U[e]=T[e]
            return U


        # ====================================================================================
        # IO for solution
        def writeSolution(self,intg,solns,stats):
            sol=self.io._createNewVolumeSolution(self,intg.runid,intg.cfg.hash(),stats['nouts'])
            for s in stats:
                sol.attrs[s]=stats[s]
            shapes=intg.rallocs.prank,{(s):solns[s].shape for s in solns}

            shapesList = self.comm.gather(shapes, root=0)
            if self.rank == self.root :
                shapes = set([ d for pr,sd in shapesList for d in sd.keys() ])
                offsets={}
                for s in shapes:
                    dataShapes=([ sd[s] for pr,sd in sorted(shapesList,key=lambda x:x[0])])
                    offsets[s]=np.cumsum((0,)+tuple([d[-1] for d in dataShapes]))
            else:
                offsets = None
            offsets=self.comm.bcast(offsets)
            
            for s in offsets:
                offset = offsets[s][intg.rallocs.prank]
                nfull  = int(offsets[s][-1])
                nspts  = int(solns[s].shape[0])
                nitems = int(solns[s].shape[1])
                neles  = int(solns[s].shape[2])
                ds=sol.create_dataset(s+"-solution",(nspts,nitems,nfull))
                ds[:,:,offset:offset+neles]=solns[s]
                del ds



class H5FileIO(IO,H5Partitioning.Partition):
    name = 'H5'
    extn = ['.pyfr']
    Partitioning = H5Partitioning
    def __init__(self,fileName,mode='r'):
        self.comm,self.rank,self.root=get_comm_rank_root()
        IO.__init__(self,fileName,mode)
        if(
                ("mesh" in self._file) and 
                ("partitionings" in self._file["mesh"]) and 
                ("__root" in self._file["mesh"]["partitionings"])
            ):
            self.__initRootPartition()

    def __initRootPartition(self):
        self.__root=self.Partitioning("__root",self)
        H5Partitioning.Partition.__init__(self,self.__root,0)
    

    # ====================================================================================
    # File operations
    def openFile(self,fileName,mode='r'):
        self._file=h5py.File(fileName,mode,driver='mpio',comm=self.comm)
        self._file.atomic=False

    def closeFile(self):
        print ("Closing file")
        self._file.close()


    # ====================================================================================
    # IO for shapes
    def writeShapes(self,**elements):
        """Create the mesh shapes"""

        assert "mesh" not in self._file
        mesh=self._file.create_group("mesh")
        shp=mesh.create_group("shape-points")
        for e in elements:
            shp.create_dataset(e,data=elements[e])

        self.group=mesh

    getShapes               = H5Partitioning.Partition.getShapes

    getShapeCounts          = H5Partitioning.Partition.getShapeCounts

    getShapePoints          = H5Partitioning.Partition.getShapePoints


    # ====================================================================================
    # IO for interfaces
    def writeInterfaces(self,interfaces):
        """Create the interfaces including the boundary interfaces"""

        assert "mesh" in self._file
        mesh=self._file["mesh"]
        intf=mesh.create_group("interfaces")
        for i in interfaces:
            if(i==0):
                ds=intf.create_dataset("internal",data=interfaces[i])
            else:
                ds=intf.create_dataset("bcon-%s"%i,data=interfaces[i])
        
        super(H5FileIO,self).writeInterfaces(interfaces)
        self.__initRootPartition()


    getConnectivities       = H5Partitioning.Partition.getConnectivities

    getInternalInterface    = H5Partitioning.Partition.getInternalInterface

    getBoundaries           = H5Partitioning.Partition.getBoundaries

    getBoundary             = H5Partitioning.Partition.getBoundary



    # ====================================================================================
    # IO for partitioning
    def writePartitioning(self,name,partitions):
        """
        Create a new partitioning with a name
        """

        assert "mesh" in self._file,"Create shapes before interfaces"

        mesh=self._file["mesh"]
        partitioning=mesh["partitionings"] if "partitionings" in mesh else  mesh.create_group("partitionings") 
        assert name not in partitioning


        partG=partitioning.create_group(name)
        nParts=len(partitions)
        partG.attrs['n-partitions']=nParts
        partG.attrs['name']=name
        parts=[partG.create_group("partition-%d"%i) for i in range(nParts)]
        if(nParts==1):
            parts[0]["interfaces"]=mesh["interfaces"]
            parts[0]["shape-points"]=mesh["shape-points"]
            el=parts[0].create_group("elements")
            for p in partitions[0]["elements"]:
                el.create_dataset(p,data=partitions[0]["elements"][p])
        else:
            for i,(gg,kk) in enumerate(zip(parts,partitions)):
                for k in kk:
                    g=gg.create_group(k)
                    if(k=="shape-points"):
                        for d in kk[k]:
                            dd = np.array(kk[k][d])
                            dd = dd.swapaxes(0,1)
                            g.create_dataset(d,data=dd)
                    elif(k=="elements"):
                        for d in kk[k]:
                            g.create_dataset(d,data=np.array(kk[k][d]))
                    elif(k=="interfaces"):
                        for d in kk[k]:
                            if(i==d):#internal partition
                                name="internal"
                            elif(type(d) is str):
                                name="bcon-%s"%d
                            else:
                                name="conn-%d"%d

                            print (type(d),":",name)
                            ifc = np.array(kk[k][d],dtype=[('type', 'S4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')]).transpose()
                            g.create_dataset(name,data=ifc)
                    else:
                        print (k)



    def getPartitionings(self):
        return self._file["mesh"]["partitionings"].keys()


    # ====================================================================================
    # IO for volume solutions
    def writeVolumeSolution(self,partition,runid,cfgid,itno):
        vol_solution=self._file["volume-solutions"] if "volume-solutions" in self._file else self._file.create_group('volume-solutions')
        sol = vol_solution.create_group("volume-solution-%s-%08d"%(runid,itno))
        sol["_partitioning"]=self._file["mesh"]["partitionings"][partition.partitioning.name]
        sol["_config"]=self._file["configs"][cfgid]
        print ("Creating solution group: ","volume-solution-%s-%08d"%(runid,itno))
        return sol

    def getSolutions(self):
        return self._file["volume-solutions"].keys()


    # ====================================================================================
    # IO for config

    def writeConfig(self,cfg):
        cfgs=self._file["configs"] if "configs" in self._file else  self._file.create_group("configs")
        data=cfg.tostr()
        if(cfg.hash() in cfgs):
            return
        ds=cfgs.create_dataset(cfg.hash(),(1,),dtype="S%s"%len(data))
        if (self.rank==self.root):
            ds[0]=data.encode('utf-8')

    def getConfigs(self):
        return self._file["configs"].keys()

    def getConfig(self,configName):
        return str(self._file["configs"][configName][0].astype('U'))




    # ====================================================================================
    # IO for code tree

    def writeCode(self):
        codes=self._file["sources"] if "sources" in self._file else  self._file.create_group("sources")
        if (self.rank==self.root):
            from .sourcetree import SourceCode
            code=SourceCode() 
            code.createArchive()
            codesize=code.size()
            codehash = code.hash()
        else:
            codesize = None
            codehash = None

        codehash=self.comm.bcast(codehash)
        print ("codehash:%s"%codehash)

        if(codehash in codes):
            return 
        
        codesize=self.comm.bcast(codesize)
        print ("codesize:%dkB"%(codesize/1024))

        ds=codes.create_dataset(codehash,(1,),dtype="S%s"%codesize)

        if (self.rank==self.root):
            ds[0]=code.raw()

    # ====================================================================================
    # Helpers 
    def _createNewVolumeSolution(self,partition,runid,cfgid,itno):
        vol_solution=self._file["volume-solutions"] if "volume-solutions" in self._file else self._file.create_group('volume-solutions')
        sol = vol_solution.create_group("volume-solution-%s-%08d"%(runid,itno))
        sol["_partitioning"]=self._file["mesh"]["partitionings"][partition.partitioning.name]
        sol["_config"]=self._file["configs"][cfgid]
        print ("Creating solution group: ","volume-solution-%s-%08d"%(runid,itno))
        return sol
        
    class Solution(IO.Solution):
        def __init__(self,name,io):
            super(self.__class__,self).__init__(name,io)
            self.io = io
            self._sol = self.io._file["volume-solutions"][name]
            self.partitioningName =  self._sol['_partitioning'].attrs['name']
            self.partitioning = self.io.getPartitioning(self.partitioningName)

        
        def getConfig(self):
            return self._sol["_config"][0].astype("U")

        def getValue(self):
            v={}
            for s in self.io.getShapes():
                shp=self.partitioning.getCompositeShapePoints(s)
                v[s]=(shp,np.array(self._sol[s+"-solution"]))
            return v
