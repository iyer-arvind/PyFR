
from abc import ABCMeta,abstractmethod
import numpy as np

class IO(object,metaclass = ABCMeta):
    def __init__(self,fileName,mode='r'):
        self.openFile(fileName,mode)


    @abstractmethod
    def openFile(self,fileName,mode='r'):
        pass

    @abstractmethod
    def closeFile(self):
        pass

    @abstractmethod
    def createShapes(self,**elements):
        pass

    @abstractmethod
    def createInterfaces(self,interfaces):
        self.__createRootPartition()

    def __createRootPartition(self):
        sc=self.getShapeCounts()
        elements={s:np.arange(sc[s]) for s in sc}
        interfaces={b:self.getBoundary(b) for b in self.getBoundaries()}
        interfaces[0]=self.getInternalInterface()
        spts={i:self.getShapePoints(i) for i in self.getShapes()}

        self.createPartitioning("__root",[{"elements":elements,"interfaces":interfaces,"shape-points":spts}])
        
    @abstractmethod    
    def createPartitioning(self,name,partitions):
        pass

    @abstractmethod
    def setVolumeSolution(self,partition,name,solutionDict,**attrs):
        pass

    @abstractmethod
    def getShapes(self):
        pass

    @abstractmethod
    def getShapeCounts(self):
        pass

    @abstractmethod
    def getShapePoints(self,shape):
        pass

    def getPartitioning(self,name):
        return self.Partitioning(name,self)

    @abstractmethod
    def getPartitionings(self):
        return self.Partitioning(name,self)

    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.closeFile()
        return False

    class Partitioning(object,metaclass=ABCMeta):
        def __init__(self,name,io):
            self.name=name
            self.io=io
            self.__nPartitions=self.nPartitions()
            self.partitions=[self.Partition(self,i) for i in range(self.__nPartitions)]

        @abstractmethod
        def nPartitions(self):
            pass

        def getConnectivities(self):
            return [p.getConnectivities() for p in self.partitions]

        def getPartition(self,part):
            return self.partitions[part]
    

        class Partition(object,metaclass=ABCMeta):
            def __init__(self,partitioning,partition):
                self.partitioning=partitioning
                self.partition=partition
                self.name=self.partitioning.name+"/partition-%d"%partition
                self.io=self.partitioning.io
            
            @abstractmethod
            def getShapes(self):
                pass

            @abstractmethod
            def getShapeCounts(self):
                pass

            @abstractmethod
            def getConnectivities(self):
                pass
            
            @abstractmethod
            def getBoundaries(self):
                pass

            @abstractmethod
            def getShapePoints(self,shape):
                pass

            @abstractmethod
            def getConnectivity(self,to):
                pass

            @abstractmethod
            def getBoundary(self,bc):
                pass
            
            @abstractmethod
            def getInternalInterface(self):
                pass

        
    
        

import h5py
class H5Partitioning(IO.Partitioning):
    def __init__(self,name,io):
        self.group=io._file["mesh"]["partitionings"][name]
        super(self.__class__,self).__init__(name,io)
    
    def nPartitions(self):
        return len(self.group.keys())

    class Partition(IO.Partitioning.Partition):
        def __init__(self,partitioning,number):
            super(H5Partitioning.Partition,self).__init__(partitioning,number)
            self.group=self.partitioning.group["partition-%d"%number]

        def getShapes(self):
            """Return the names of the shapes"""
            return self.group["shape-points"].keys()

        def getShapeCounts(self):
            """Return a dict of the shapes and the count of the elements of that shape"""
            shp=self.group["shape-points"]
            return {s:shp[s].shape[1] for s in shp}

        def getConnectivities(self):
            return tuple(map(lambda x:int(x.replace('conn-',"")),filter(lambda x:x.startswith("conn-"),self.group["interfaces"].keys())))
        
        def getBoundaries(self):
            return list(map(lambda x:x.replace("bcon-",""),filter(lambda x:x.startswith("bcon-"),self.group["interfaces"].keys())))

        def getShapePoints(self,shape):
            return np.array(self.group["shape-points"][shape])



        def getConnectivity(self,to):
            return self.__getInterface("conn-%d"%to)

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

class H5FileIO(IO,H5Partitioning.Partition):

    Partitioning = H5Partitioning

    def __init__(self,fileName,mode='r'):
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

    def openFile(self,fileName,mode='r'):
        self._file=h5py.File(fileName,mode)

    def closeFile(self):
        self._file.close()

    def createShapes(self,**elements):
        """Create the mesh shapes"""

        assert "mesh" not in self._file
        mesh=self._file.create_group("mesh")
        shp=mesh.create_group("shape-points")
        for e in elements:
            shp.create_dataset(e,data=elements[e])

        self.group=mesh

    def createInterfaces(self,interfaces):
        """Create the interfaces including the boundary interfaces"""

        assert "mesh" in self._file
        mesh=self._file["mesh"]
        intf=mesh.create_group("interfaces")
        for i in interfaces:
            if(i==0):
                ds=intf.create_dataset("internal",data=interfaces[i])
            else:
                ds=intf.create_dataset("bcon-%s"%i,data=interfaces[i])
        
        super(H5FileIO,self).createInterfaces(interfaces)
        self.__initRootPartition()


    def createPartitioning(self,name,partitions):
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

    getConnectivities       = H5Partitioning.Partition.getConnectivities
    getConnectivity         = H5Partitioning.Partition.getConnectivity

    getInternalInterface    = H5Partitioning.Partition.getInternalInterface
    getShapes               = H5Partitioning.Partition.getShapes
    getShapePoints          = H5Partitioning.Partition.getShapePoints
    getShapeCounts          = H5Partitioning.Partition.getShapeCounts

    getBoundaries           = H5Partitioning.Partition.getBoundaries
    getBoundary             = H5Partitioning.Partition.getBoundary


    def getPartitionings(self):
        return self._file["mesh"]["partitionings"].keys()

    def setVolumeSolution(self,partition,name,solutionDict,**attrs):
        sols=self._file["volume-solutions"] if "volume-solutions" in self._file else  self._file.create_group("volume-solutions")
        sol=sols.create_group(name)
        for a in attrs:
            sol.attrs[a]=attrs[a]
        for s in solutionDict:
            shapeSol=sol.create_group(s)
            for ss in solutionDict[s]:
                shapeSol.create_dataset(ss,data=solutionDict[s][ss])



        
