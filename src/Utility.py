#  Copyright (c) 2023.
#  Mohammed Kharma
'''Utility class'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Utility:
    '''
    Discribe the dataframe
    '''

    def describeDS(self, originalDS):
        print("Dataset information:")
        print(originalDS.info())
        print("Dataset description:")
        print(originalDS.describe())
        print("Dataset sample:")
        print(originalDS.head())

    '''
    Read dataset from file
    '''

    def readDataSet(self, filename):
        # read the DS as Pandas df
        originalDS = pd.read_csv(filename, sep=",", header=None)
        # originalDS = pd.read_csv("../dataset/3gaussians/3gaussians-std0.9.csv", sep=",", header=None)
        # originalDS = pd.read_csv("../dataset/Iris/iris.data", sep=",", header=None)
        return originalDS

    '''Draw the data points along with the clusters centers'''

    def plot(self, clusterCenters, clusterMemberships, workingCopyDS,algorithmConvergeHistory):
        x_index = 0
        y_index = 1
        # this formatter will label the colorbar with the correct target names
        # formatter = plt.FuncFormatter(lambda i, *args: labels[int(i)])
        plt.figure(figsize=(5, 4))
        plt.scatter(workingCopyDS[:, x_index], workingCopyDS[:, y_index],alpha=0.4,
                    c=clusterMemberships)
        plt.scatter(clusterCenters[:, 0], clusterCenters[:, 1],alpha=0.4, s=100, color='y')
        # c=originalDS.to_numpy(copy=True)[:, y_index+1])
        # plt.colorbar(ticks=[0, 1, 2], format=formatter)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

        sns.lineplot(x=range(0,len(algorithmConvergeHistory)), y=algorithmConvergeHistory, marker='x')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()