#  Copyright (c) 2023.
#  Mohammed Kharma
# This class represent the K-means algorithm implementation

import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class KmeansAlgorithm:
    '''
    The K-means algorithm implementation function

    originalDS: Pandas datafram
    reclassifcationIterationLimit: Limit number of algorithm iteration
    stopLimit: break the algorithm iteration upon the specified error rate/conversion point
    numberOfClusters: number of clusters to be created,
                        If NONE, then get number of columns except the label column.
    needDimReduction: Applying dimensionality reduction before running k-means can help to
                    reduce the computational cost and improve the performance of the algorithm,
                    as well as to identify patterns or relationships in the data that may not be easily
                    visible in higher dimensions.
    '''

    def kmeansAlgorithmPureImpl(_self, originalDS,
                                reclassifcationIterationLimit,
                                stopLimit=0.01,
                                numberOfClusters=None,
                                needDimReduction=False):

        # Number of rows in the DS
        global prevcostFunction
        global costFunction
        prevcostFunction = 0
        numberOfPoints = len(originalDS.index)

        # Number of columns in the DS including the label
        numberOfColumnsInDS = len(originalDS.columns)

        # All DS has feature columns except the last column as its the label of the instance
        # Number of features is count of columns without the label column
        # Since we are using dimintiality reduction we always will deal with 2D
        numberOfFeatures = 2
        # In all DS the label column index is the last column
        indexOfLabelColumn = numberOfColumnsInDS - 1

        # If no desired number of cluster is provided, then get it from the DS
        if numberOfClusters is None:
            numberOfClusters = originalDS[originalDS.columns[indexOfLabelColumn]].unique().size

        # remove the last column in the DS
        workingCopyDS = originalDS.iloc[:,
                        np.r_[range(indexOfLabelColumn), range(indexOfLabelColumn + 1, len(originalDS.columns))]]

        # Prepare the data as np DS for better support mathematical operations later on
        workingCopyDS = workingCopyDS.to_numpy(copy=True)

        # Applying dimensionality reduction before running k-means can help to reduce
        # the computational cost and improve the performance of the algorithm,
        # as well as to identify patterns or relationships
        # in the data that may not be easily visible in higher dimensions.
        if needDimReduction:
            pca = PCA(n_components=2)
            workingCopyDS = pca.fit_transform(workingCopyDS)
            ## print("Dataset after dimensionality reduction: ", workingCopyDS)

        # Initialize the scaler in order to use Min-Max normalization
        scaler = MinMaxScaler()
        # Fit and transform the data
        workingCopyDS = scaler.fit_transform(workingCopyDS)

        # Fix the randomization behaviour for better testing and results comparison
        # np.random.seed(6)
        # Before we start the clustering, the first step is to initialize the cluster centers
        # based on number of the selected clusters by selecting random points from the DS.
        idx = np.random.randint(numberOfPoints, size=numberOfClusters)
        clusterCenters = workingCopyDS[idx, :]

        ## print("Initial cluster centers: ", clusterCenters)

        # Initialize two dim zeros array with length of the clusters X points in the DS
        clusterMemberships = np.empty(numberOfPoints)
        # clusterMemberships1 = [[0]*numberOfPoints for _ in range(numberOfClusters)]
        # [[] for _ in range(numberOfClusters)]
        # The stop condition for the cluster center calculation iteration when the function reach to
        # the convergent point or end of the iteration count
        isClusterCenterGetConverged = False

        # Store the history of the changes in the centers over the iterations
        algorithmConvergeHistory = []

        # Repeat until reach to the iteration limit
        for currentIteration in range(reclassifcationIterationLimit):
            # Stop if the convergent point met
            if isClusterCenterGetConverged:
                break

            ## print("Current iteration #: ", currentIteration)

            # For each point in the DS, find the Euclidean distance and assign it to the
            # closest cluster center
            for pointInstanceIndex in range(numberOfPoints):
                currentPoint = workingCopyDS[pointInstanceIndex]

                # Set the smallest found distance to a large number
                bestDistance = sys.maxsize

                # Find which cluster center is the closest one to the current point using
                # calculating the Euclidean distance
                for clusterCenterIndex in range(numberOfClusters):
                    currentClusterCenter = clusterCenters[clusterCenterIndex]

                    distance = euclideanDistance(currentClusterCenter, currentPoint)

                    # If the calculated distance is smaller than the best found distance,
                    # then consider it and asign this best cluster center to the current point by
                    # updating the assigned cluster center to the current point.
                    # And then update the best found distance for the next iteration
                    if distance < bestDistance:
                        clusterMemberships[pointInstanceIndex] = clusterCenterIndex
                        # Update the index to one and the same column index in the other
                        # clusters will remain as zero as per to the array initialization
                        # clusterMemberships1[clusterCenterIndex][pointInstanceIndex]=1

                        bestDistance = distance

            # points assignments on each cluster result

            ## print("Points assignments on each cluster result for iteration # ", currentIteration,
            ##      ": ", clusterMemberships)
            # print("Points assignments on each cluster result for iteration # ",currentIteration,
            #       ": ", clusterMemberships1)

            # After the above round of asigning the closest cluster to each point,
            # we need to move the clsueter center based on the new assigned points mean
            # to each cluster

            # Take a copy of last found cluster centers, then to compare
            # if the new calculated cluster center covergent.
            lastKnownCluseterCenters = clusterCenters.copy()

            # For each cluster center, calculate the new ceneter based on the
            # mean of the assigned points to that cluster
            for clusterCenterIndex in range(numberOfClusters):

                # temporary array to store the sum of the point that related to the current
                # cluster.
                sumOfPoints = [0.0] * numberOfFeatures
                for i in range(numberOfFeatures):
                    # Set the sum of all array element to zero
                    sumOfPoints[i] = 0
                numberOfAssignedClusterPoints = 0
                # sumOfPoints1 = [0.0] * numberOfFeatures
                # for i in range(numberOfFeatures):
                #     #Set the sum of all array element to zero
                #     sumOfPoints1[i] = 0
                # numberOfAssignedClusterPoints1 = 0

                # Find the sum of all points of the current cluster
                for pointInstanceIndex in range(numberOfPoints):

                    # If this instance belongs to this cluster
                    if (clusterCenterIndex == clusterMemberships[pointInstanceIndex]):
                        # Increase the sumOfPoints by the new assigned point
                        sumOfPoints = sumOfPoints + \
                                      workingCopyDS[pointInstanceIndex]
                        # _self.euclideanDistance(workingCopyDS[pointInstanceIndex],
                        #                                     lastKnownCluseterCenters)

                        # Increase the number of the assigned point to the current cluster by 1
                        numberOfAssignedClusterPoints = numberOfAssignedClusterPoints + 1

                    # #Old: if value is 1 then its in the current cluster
                    # if(clusterMemberships1[clusterCenterIndex][pointInstanceIndex] == 1):
                    #     # Increase the sumOfPoints by the new assigned point
                    #     sumOfPoints1 = sumOfPoints1 + workingCopyDS[pointInstanceIndex]
                    #
                    #     # Increase the number of the assigned point to the current cluster by 1
                    #     numberOfAssignedClusterPoints1 = numberOfAssignedClusterPoints1 + 1
                    #

                # Consider the mean of the current cluster assigned points as
                # the new cluster center
                # print ("sumOfPoints-----", sumOfPoints, "numberOfAssignedClusterPoints--", numberOfAssignedClusterPoints)
                if numberOfAssignedClusterPoints != 0:
                    clusterCenters[clusterCenterIndex] = sumOfPoints / numberOfAssignedClusterPoints
                else:
                    clusterCenters[clusterCenterIndex] = 0
                #
                # print("New clacluated clusters in iteration # ",currentIteration,
                #       ": ", sumOfPoints1/numberOfAssignedClusterPoints1)

                ##print("New clacluated clusters in iteration # ", currentIteration,
            ##     ": ", clusterCenters)

            # sumOfCostFunctionForEachCluseterCenters = clusterCenters.copy()
            # Calculate the cost function
            numberOfAssignedClusterPoints = 0
            # temporary array to store the sum of the point that related to the current
            # cluster.
            sumOfCostFunction = 0.0

            for clusterCenterIndex in range(numberOfClusters):

                # Find the sum of all points of the current cluster
                for pointInstanceIndex in range(numberOfPoints):

                    # If this instance belongs to this cluster
                    if (clusterCenterIndex == clusterMemberships[pointInstanceIndex]):
                        # Increase the sumOfPoints by the new assigned point
                        sumOfCostFunction = sumOfCostFunction + \
                                            euclideanDistanceWithSum(workingCopyDS[pointInstanceIndex],
                                                                     clusterCenters[clusterCenterIndex])

                        # Increase the number of the assigned point to the current cluster by 1
                        numberOfAssignedClusterPoints = numberOfAssignedClusterPoints + 1

            # Consider the mean of the current cluster assigned points as
            # the new cluster center
            # sumOfCostFunctionForEachCluseterCenters[clusterCenterIndex] = sumOfCostFunction/numberOfAssignedClusterPoints

            costFunction = sumOfCostFunction / numberOfAssignedClusterPoints
            currentCostFunctionDiff = abs(costFunction - prevcostFunction)
            prevcostFunction = costFunction
            if currentCostFunctionDiff <= stopLimit:
                isClusterCenterGetConverged = True

            ## print("algorithmConvergeHistory: ", algorithmConvergeHistory)
            algorithmConvergeHistory.append(costFunction)
        return workingCopyDS, clusterMemberships, clusterCenters, \
               algorithmConvergeHistory, costFunction


def euclideanDistance(currentClusterCenter, currentPoint):
    # Find the differences between the current point's features and the
    # current cluster center features
    difference = currentPoint - currentClusterCenter
    # Calculate the square the difference
    squaredDifference = difference ** 2
    # Take the summation of the squared differences
    SumSquaredDifference = np.sum(squaredDifference)
    # Take the square root of the summation of the squared differences
    distance = np.sqrt(SumSquaredDifference)
    return distance


def euclideanDistanceWithSum(currentClusterCenter, currentPoint):
    # Find the differences between the current point's features and the
    # current cluster center features
    difference = currentPoint - currentClusterCenter
    # Calculate the square the difference
    squaredDifference = difference ** 2
    # Take the summation of the squared differences
    SumSquaredDifference = np.sum(squaredDifference)
    # Take the square root of the summation of the squared differences
    distance = np.sum(np.sqrt(SumSquaredDifference))
    return distance
