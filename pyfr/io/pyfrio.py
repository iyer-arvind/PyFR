
from abc import ABCMeta,abstractmethod
import numpy as np

class BaseIO(object,metaclass = ABCMeta):
    def __init__(self,fileName,mode='r'):
        self.openFile(fileName,mode)


    def __createRootPartition(self):
        sc=self.getShapeCounts()
        elements={s:np.arange(sc[s]) for s in sc}
        interfaces={b:self.getBoundary(b) for b in self.getBoundaries()}
        interfaces[0]=self.getInternalInterface()
        spts={i:self.getShapePoints(i) for i in self.getShapes()}

        self.writePartitioning("__root",[{"elements":elements,"interfaces":interfaces,"shape-points":spts}])


    # Support for the _with_ statement
    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.closeFile()
        return False

    # ====================================================================================
    # File Operations
    @abstractmethod
    def openFile(self,fileName,mode='r'):
        pass

    @abstractmethod
    def closeFile(self):
        pass

     
    # ====================================================================================
    # IO for shapes
    @abstractmethod
    def writeShapes(self,**elements):
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

    # ====================================================================================
    # IO for Interfaces
    @abstractmethod
    def writeInterfaces(self,interfaces):
        self.__createRootPartition()

    @abstractmethod
    def getInternalInterface(self):
        pass


    @abstractmethod
    def getInternalInterface(self):
        pass

    @abstractmethod
    def getBoundaries(self):
        pass

    @abstractmethod
    def getBoundary(self,bc):
        pass


    # ====================================================================================
    # IO for partitioning
    @abstractmethod    
    def writePartitioning(self,name,partitions):
        pass

    @abstractmethod
    def getPartitionings(self):
        pass

    def getPartitioning(self,name):
        return self.Partitioning(name,self)

    # ====================================================================================
    # IO for volume solutions
    @abstractmethod    
    def writeVolumeSolution(self,partition,runid,cfgid,itno):
        pass

    @abstractmethod
    def getSolutions(self):
        pass

    def getSolution(self,solutionName):
        return self.Solution(solutionName,self)


    # ====================================================================================
    # IO for config
    @abstractmethod    
    def writeConfig(self,cfg):
        pass

    @abstractmethod
    def getConfigs(self):
        pass

    @abstractmethod
    def getConfig(self,configName):
        pass








    #######################################################################################
    ########### PARTITIONING CLASS
    class Partitioning(object,metaclass=ABCMeta):
        def __init__(self,name,io):
            self.name=name
            self.io=io
            self.__nPartitions=self.nPartitions()
            self.partitions=[self.Partition(self,i) for i in range(self.__nPartitions)]


        # number of partitions
        @abstractmethod
        def nPartitions(self):
            pass

        def getConnectivities(self):
            return [p.getConnectivities() for p in self.partitions]

        def getPartition(self,part):
            return self.partitions[part]
    
        #######################################################################################
        ########### PARTITION CLASS
        class Partition(object,metaclass=ABCMeta):
            def __init__(self,partitioning,partition):
                self.partitioning=partitioning
                self.partition=partition
                self.name=self.partitioning.name+"/partition-%d"%partition
                self.io=self.partitioning.io
            
            # ====================================================================================
            # IO for shapes
            @abstractmethod
            def getShapes(self):
                pass

            @abstractmethod
            def getShapeCounts(self):
                pass

            @abstractmethod
            def getShapePoints(self,shape):
                pass

            # ====================================================================================
            # IO for internal interfaces, boundaries and connectivity
            @abstractmethod
            def getConnectivities(self):
                pass

            @abstractmethod
            def getConnectivity(self,to):
                pass
            
            @abstractmethod
            def getBoundaries(self):
                pass

            @abstractmethod
            def getBoundary(self,bc):
                pass

            @abstractmethod
            def getInternalInterface(self):
                pass


            # ====================================================================================
            # IO for solution
            @abstractmethod
            def writeSolution(self,solns,stats):
                pass

        

    #######################################################################################
    ########### Solution class
    class Solution(object,metaclass=ABCMeta):
        def __init__(self,name,io):
            self.name = name
            self.iop = io

        @abstractmethod
        def getConfig(self):
            pass


        @abstractmethod
        def getValue(self):
            pass



        
    
        


        
        

            


