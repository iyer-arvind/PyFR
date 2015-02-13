
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
        partition={s:[np.arange(sc[s])] for s in sc}
        self.createPartitioning('root',partition)
        
    @abstractmethod    
    def createPartitioning(self,name,part):
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

    def getPartitioning(self,name):
        return self.Partitioning(name,self)

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


    def createPartitioning(self,name,part):
        """Create a new partitioning with a name"""

        assert "mesh" in self._file,"Create shapes before interfaces"
        mesh=self._file["mesh"]
        partitioning=mesh["partitioning"] if "partitioning" in mesh else  mesh.create_group("partitioning") 
        assert name not in partitioning
        partG=partitioning.create_group(name)
        for s in part:
            for pi,p in enumerate(part[s]):
                partG.create_dataset(s+"-%d"%pi,data=p)
        partG.attrs['n-partitions']=len(list(part.values())[0])

    def getPartitionsInPartitioning(self,partitioning):
        return self._file["mesh"]["partitioning"][partitioning].attrs["n-partitions"]


    def setVolumeSolution(self,partition,name,solutionDict,**attrs):
        sols=self._file["volume-solutions"] if "volume-solutions" in self._file else  self._file.create_group("volume-solutions")
        sol=sols.create_group(name)
        for a in attrs:
            sol.attrs[a]=attrs[a]
        for s in solutionDict:
            shapeSol=sol.create_group(s)
            for ss in solutionDict[s]:
                shapeSol.create_dataset(ss,data=solutionDict[s][ss])
            

    def shapes(self):
        """Return the names of the shapes"""

        return self._file["mesh"]["shape-points"].keys()

    def shapeCounts(self):
        """Return a dict of the shapes and the count of the elements of that shape"""

        shp=self._file["mesh"]["shape-points"]
        return {s:shp[s].shape[1] for s in shp}
    

        
