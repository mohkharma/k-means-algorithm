#  Copyright (c) 2023.
#  Mohammed Kharma
'''
    Algorithm testing .
'''

from src.KmeansAlgorithmV3 import KmeansAlgorithm
from src.Utility import Utility


class AlgorithmsTesting:
    def testKmeansPureImpl(self):
        '''
        Read dataset from file
        '''
        utility = Utility()
        needDimReduction = False
        # originalDS = utility.readDataSet("../dataset/circles.csv")
        # originalDS = utility.readDataSet("../dataset/moons.csv")
        # originalDS = utility.readDataSet("../dataset/3gaussians/3gaussians-std0.9.csv")
        # originalDS = utility.readDataSet("../dataset/3gaussians/3gaussians-std0.6.csv")
        originalDS = utility.readDataSet("../dataset/Iris/iris.data")
        needDimReduction = True
        numberOfClusters = 3
        # Describe the data
        utility.describeDS(originalDS)

        kmeansAlgorithm = KmeansAlgorithm()
        # Call the K-mean function
        workingCopyDS, clusterMemberships, clusterCenters, algorithmConvergeHistory = \
            kmeansAlgorithm.kmeansAlgorithmPureImpl(originalDS=originalDS,
                                                    reclassifcationIterationLimit=200,
                                                    stopLimit=0.2,
                                                    numberOfClusters = numberOfClusters,
                                                    needDimReduction=needDimReduction)

        # Draw the data points along with the clusters centers                                          )
        utility.plot(clusterCenters, clusterMemberships, workingCopyDS,algorithmConvergeHistory)
        #,originalDS[originalDS.columns[len(originalDS.columns)-1]].unique()

algorithmsTesting = AlgorithmsTesting()

algorithmsTesting.testKmeansPureImpl()
