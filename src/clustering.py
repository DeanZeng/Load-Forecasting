#! /usr/bin/python

import math
import statistics
import visualizer
import numpy as np
from datagen import constructData
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Ward

# Applies clustering based algorithm to the electricity dataset,
# prints out the accuracy rate to the terminal and plots
# predictions against actual values
def clustering():

    # Retrieve time series data & apply preprocessing
    data = constructData()

    # 2014 had 365 days, but we take the last 364 days since
    # the last day has no numerical value
    cutoff = len(data)-364
    xTrain = data[0][0:cutoff]
    yTrain = data[1][0:cutoff]
    xTest = data[0][cutoff:]
    yTest = data[1][cutoff:]

    # Fill in missing values denoted by zeroes as an average of
    # both neighbors
    statistics.estimateMissing(xTrain,0.0)
    statistics.estimateMissing(xTest,0.0)

    # Logarithmically scale the data
    xTrain = [[math.log(y) for y in x] for x in xTrain]
    xTest = [[math.log(y) for y in x] for x in xTest]
    yTrain = [math.log(x) for x in yTrain]

    # Detrend the time series
    indices = np.arange(len(data[1]))
    trainIndices = indices[0:cutoff]
    testIndices = indices[cutoff:]
    detrended,slope,intercept = statistics.detrend(trainIndices,yTrain)
    yTrain = detrended

    # Compute centroids and labels of data
    cward_7,lward_7 = hierarchicalClustering(xTrain,7)
    cward_365,lward_365 = hierarchicalClustering(xTrain,365)

    ckmeans_7,lkmeans_7 = kMeansClustering(xTrain,7)
    ckmeans_365,lkmeans_365 = kMeansClustering(xTrain,365)

    c = [cward_7,cward_365,ckmeans_7,ckmeans_365]
    l = [lward_7,lward_365,lkmeans_7,lkmeans_365]

    algNames = ["agglomerative(7)","agglomerative(365)","k-means(7)","k-means(365)"]

    preds = []

    for t in range(len(c)):
        # The centroids computed by the current clustering algorithm
        centroids = c[t]
        # The labels for the examples defined by the current clustering assignment
        labels = l[t]

        # Separate the training samples into cluster sets
        clusterSets = []
        # Time labels for the examples, separated into clusters
        timeLabels = []

        for x in range(len(centroids)):
            clusterSets.append([])
        for x in range(len(labels)):
            # Place the example into its cluster
            clusterSets[labels[x]].append((xTrain[x],yTrain[x]))
        # Compute predictions for each of the test examples
        pred = predictClustering(centroids,clusterSets,xTest,"euclidean")
        # Add the trend back into the predictions
        trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
        # Reverse the normalization
        trendedPred = [math.exp(x) for x in trendedPred]
        # Compute the NRMSE
        err = statistics.normRmse(yTest,trendedPred)
        # Add to list of predictions
        preds.append(trendedPred)

        print "The Normalized Root-Mean Square Error is " + str(err) + " using algorithm " + algNames[t] + "..."

    algNames.append("actual")
    preds.append(yTest)

    visualizer.comparisonPlot(2014,1,1,preds,algNames, 
        plotName="Clustering Load Predictions vs. Actual", 
        yAxisName="Predicted Kilowatts")

# Performs Hierarchical Clustering using Ward Linkage
def hierarchicalClustering(x,k):
    model = Ward(n_clusters=k)
    labels = model.fit_predict(np.asarray(x))

    # Centroids is a list of lists
    centroids = []
    for c in range(k):
        base = []
        for d in range(len(x[0])):
            base.append(0)
        centroids.append(base)

    # Stores number of examples per cluster
    ctrs = np.zeros(k)

    # Sum up all vectors for each cluster
    for c in range(len(x)):
        centDex = labels[c]
        for d in range(len(centroids[centDex])):
            centroids[centDex][d] += x[c][d]
        ctrs[centDex] += 1

    # Average the vectors in each cluster to get the centroids
    for c in range(len(centroids)):
        for d in range(len(centroids[c])):
            centroids[c][d] = centroids[c][d]/ctrs[c]

    return (centroids,labels)

# Performs K-Means Clustering on the ordered sequence
# of vectors x with parameter k, and returns a 2-tuple:
# First tuple value is list of centroids
# Second tuple value is vector x' of length equal to that
# of x, such that the ith 
# value of x' is the cluster label for the ith example
# of the input x
def kMeansClustering(x,k):

    # Convert list into numpy format
    conv = np.asarray(x)

    # Compute the centroids
    centroids = kmeans(conv,k,iter=10)[0]

    # Relabel the x's
    labels = []
    for y in range(len(x)):
        minDist = float('inf')
        minLabel = -1
        for z in range(len(centroids)):
            e = euclidean(conv[y],centroids[z])
            if (e < minDist):
                minDist = e
                minLabel = z
        labels.append(minLabel)

    # Return the list of centroids and labels
    return (centroids,labels)

# Performs a weighted clustering on the examples in xTest
# Returns a 1-d vector of predictions
def predictClustering(clusters,clusterSets,xTest,metric):
    clustLabels = []
    simFunction = getDistLambda(metric)
    for x in range(len(xTest)):
        clustDex = -1
        clustDist = float('inf')
        for y in range(len(clusters)):
            dist = simFunction(clusters[y],xTest[x])
            if (dist < clustDist):
                clustDist = dist
                clustDex = y
        clustLabels.append(clustDex)
    predict = np.zeros(len(xTest))
    for x in range(len(xTest)):
        predict[x] = weightedClusterClass(xTest[x],clusterSets[clustLabels[x]],simFunction)
    return predict

# Performs a weighted cluster classification
def weightedClusterClass(xVector,examples,simFunction):
    pred = 0.0
    normalizer = 0.0
    ctr = 0
    for x in examples:
        similarity = 1.0/simFunction(xVector,x[0])
        pred += similarity*x[1]
        normalizer += similarity
        ctr += 1
    return (pred/normalizer)

def getDistLambda(metric):
    if (metric == "manhattan"):
        return lambda x,y : distance.cityblock(x,y)
    elif (metric == "cosine"):
        return lambda x,y : distance.cosine(x,y)
    else:
        return lambda x,y : distance.euclidean(x,y)

if __name__=="__main__":
    clustering()