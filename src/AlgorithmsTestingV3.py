#  Copyright (c) 2023.
#  Mohammed Kharma
'''
    Algorithm testing .
'''

import time

from src.KmeansAlgorithmV3 import KmeansAlgorithm
from src.Utility import Utility
import matplotlib.pyplot as plt
import seaborn as sns

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
        originalDS = utility.readDataSet("../dataset/3gaussians/3gaussians-std0.6.csv")
        # originalDS = utility.readDataSet("../dataset/Iris/iris.data")
        # // this to be used for  Iris
        # needDimReduction = True

        bestCostFunction = 999999999
        bestNumberOfClusters = 99999999
        bestWorkingCopyDS = None
        bestClusterMemberships = None
        bestClusterCenters = None
        clusterBasedAlgorithmConvergeHistory = []

        # Describe the data
        utility.describeDS(originalDS)

        kmeansAlgorithm = KmeansAlgorithm()
        # Call the K-mean function
        maxNumberOfClusters = 4
        prevoiusBestRunCostFunction = 999999999
        t = time.localtime()
        for numberOfClusters in range(2, maxNumberOfClusters):
            workingCopyDS, clusterMemberships, clusterCenters, algorithmConvergeHistory, \
            costFunction = \
                kmeansAlgorithm.kmeansAlgorithmPureImpl(originalDS=originalDS,
                                                        reclassifcationIterationLimit=300,
                                                        stopLimit=0.000000000001,
                                                        numberOfClusters=numberOfClusters,
                                                        needDimReduction=needDimReduction)

            # clusterBasedAlgorithmConvergeHistory.append(costFunction)
            # if bestCostFunction > costFunction:
            #     bestNumberOfClusters = numberOfClusters
            #     bestAlgorithmConvergeHistory = algorithmConvergeHistory
            #     bestWorkingCopyDS = workingCopyDS
            #     bestClusterMemberships = clusterMemberships
            #     bestClusterCenters = clusterCenters
            #     bestCostFunction = costFunction

            #     If no significant improvement, then break
            # if prevoiusBestRunCostFunction - costFunction <= 0.0002:
            #     break
            # prevoiusBestRunCostFunction = costFunction
            # sns.lineplot(x=range(0 + 1, len(algorithmConvergeHistory) + 1), y=algorithmConvergeHistory, marker='x')

            utility.plot(clusterCenters, clusterMemberships, workingCopyDS,
                         algorithmConvergeHistory)

        print("Time in execution: ", (time.localtime().tm_sec - t.tm_sec))
        # Draw the data points along with the clusters centers                                          )
        # utility.plot(bestClusterCenters, bestClusterMemberships, bestWorkingCopyDS,
        #              clusterBasedAlgorithmConvergeHistory)
        # ,originalDS[originalDS.columns[len(originalDS.columns)-1]].unique()

algorithmsTesting = AlgorithmsTesting()

algorithmsTesting.testKmeansPureImpl()
