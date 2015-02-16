
from abc import ABCMeta,abstractmethod
import numpy as np

class IO(object):
    __metaclass__ = ABCMeta
    def __init__(self,fileName,mode='r'):
        super(IO,self).__init__()
        self.open(fileName,mode)


    @abstractmethod
    def open(self,fileName,mode='r'):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def createShapes(self,**elements):
        pass

    @abstractmethod
    def createInterfaces(self,**elements):
        self.__createDefaultPartition()

    def __createDefaultPartition(self):
        sc=self.shapeCounts()
        elements={s+"-0":np.arange(sc[s]) for s in sc}
        interfaces={ i+'-0':self.getInterface(i) for i in self.getInterfaces()}
        spts={i+'-0':self.getShapePoints(i) for i in self.shapes()}
        self.createPartitioning('root',elements,interfaces,spts)
        
    @abstractmethod    
    def createPartitioning(self,name,elements,interfaces):
        pass

    @abstractmethod
    def setVolumeSolution(self,partition,name,solutionDict,**attrs):
        pass

    @abstractmethod
    def shapes(self):
        pass

    @abstractmethod
    def shapeCounts(self):
        pass

    @abstractmethod
    def getShapePoints(self,shape):
        pass

    def getPartitioning(self,name):
        return self.Partitioning(name,self)

    @abstractmethod
    def getPartitionings(self):
        return self.Partitioning(name,self)


    @abstractmethod
    def getInterface(self,name):
        pass

    @abstractmethod
    def getInterfaces(self):
        pass

    @abstractmethod
    def getPartitionsInPartitioning(self,partitioning):
        pass

    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.close()
        return False

    class Partitioning(object):
        def __init__(self,name,io):
            self.name=name
            self.io=io
            self.__nPartitions=io.getPartitionsInPartitioning(self.name)


        def getPartition(self,part):
            assert part<self.__nPartitions
            return self.Partition(self,part)
    

        class Partition(object):
            def __init__(self,partitioning,partition):
                self.partitioning=partitioning
                self.partition=partition
                self.io=self.partitioning.io

            def setVolumeSolution(self,name,solutionDict,**attrs):
                attrs["__partitioning"]=self.partitioning.name
                self.io.setVolumeSolution(self.partition,name,solutionDict,**attrs)

        
    
        

import h5py
class H5FileIO(IO):

    def __init__(self,*args,**kwargs):
        super(H5FileIO,self).__init__(*args,**kwargs)


    def open(self,fileName,mode='r'):
        self._file=h5py.File(fileName,mode)


    def close(self):
        self._file.close()

    def createShapes(self,**elements):
        """Create the mesh shapes"""

        assert "mesh" not in self._file
        mesh=self._file.create_group("mesh")
        shp=mesh.create_group("shape-points")
        for e in elements:
            shp.create_dataset(e,data=elements[e])


    def createInterfaces(self,**interfaces):
        """Create the interfaces including the boundary interfaces"""

        assert "mesh" in self._file
        mesh=self._file["mesh"]
        intf=mesh.create_group("interfaces")
        for i in interfaces:
            intf.create_dataset(i,data=interfaces[i])
        super(H5FileIO,self).createInterfaces()

    def getInterfaces(self):
        return self._file['mesh']['interfaces'].keys()

    def getInterface(self,name):
        T=np.array(self._file['mesh']['interfaces'][name])
        dtype=[('type', 'U4'), ('ele', '<i4'), ('face', 'i1'),('zone', 'i1')]
        U=np.zeros_like(T,dtype=dtype)
        for e,t in  dtype:
            U[e]=T[e]
        return U

    def createPartitioning(self,name,elements,interfaces,spts):
        """Create a new partitioning with a name"""

        assert "mesh" in self._file,"Create shapes before interfaces"
        mesh=self._file["mesh"]
        partitioning=mesh["partitionings"] if "partitionings" in mesh else  mesh.create_group("partitionings") 
        assert name not in partitioning

        partG=partitioning.create_group(name)
        elments=partG.create_group("elements")

        for s in elements:
            elments.create_dataset(s,data=elements[s])

        nParts=len(set([e.split('-')[1] for e in elements]))
        partG.attrs['n-partitions']=nParts
        intf=partG.create_group("interfaces")

        shp=partG.create_group("shapes")
        if(nParts==1):# This is the root partition!
            for i in interfaces:
                assert i.endswith('-0')
                intf[i]=self._file["mesh"]['interfaces'][i.replace('-0','')]

            for s in spts:
                shp[s]=self._file["mesh"]['shape-points'][s.replace('-0','')]
        else:
            for i in interfaces:
                intf.create_dataset(i,data=interfaces[i])
            for s in spts:
                shp.create_dataset(s,data=spts[s])


    

    def getPartitionsInPartitioning(self,partitioning):
        return self._file["mesh"]["partitionings"][partitioning].attrs["n-partitions"]


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

    def getMesh(self):
        pass
            
    def shapes(self):
        """Return the names of the shapes"""
        return self._file["mesh"]["shape-points"].keys()

    def shapeCounts(self):
        """Return a dict of the shapes and the count of the elements of that shape"""

        shp=self._file["mesh"]["shape-points"]
        return {s:shp[s].shape[1] for s in shp}
    
    def getShapePoints(self,shape):
        return np.array(self._file["mesh"]["shape-points"][shape])

        
