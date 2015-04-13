#! /usr/bin/python

import math
import visualizer
import statistics
import numpy as np
from datagen import constructData
from sklearn import gaussian_process

# Applies Gaussian Processes to the electricity dataset,
# prints out the accuracy rate to the terminal and plots
# predictions against actual values
def gaussianProcesses():

    corrMods = ['cubic','squared_exponential','absolute_exponential','linear']
    preds = []

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

    for gen in range(len(corrMods)):

        # Use GPR to predict test observations based upon training observations
        pred = gaussProcPred(xTrain,yTrain,xTest,corrMods[gen])
        # Add the trend back into the predictions
        trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
        # Reverse the normalization
        trendedPred = [math.exp(x) for x in trendedPred]
        # Compute the NRMSE
        err = statistics.normRmse(yTest,trendedPred)

        print "The Normalized Root-Mean Square Error is " + str(err) + " using covariance function " + corrMods[gen] + "..."

        preds.append(trendedPred)

    corrMods.append("actual")
    data = constructData()
    cutoff = len(data)-364
    yTest = data[1][cutoff:]
    preds.append(yTest)

    visualizer.comparisonPlot(2014,1,1,preds,corrMods,plotName="Gaussian Process Regression Load Predictions vs. Actual", 
        yAxisName="Predicted Kilowatts")

# Gaussian Process Regression
def gaussProcPred(xTrain,yTrain,xTest,covar):
    xTrainAlter = []
    for i in range(1,len(xTrain)):
        tvec = xTrain[i-1]+xTrain[i]
        xTrainAlter.append(tvec)
    xTestAlter = []
    xTestAlter.append(xTrain[len(xTrain)-1]+xTest[0])
    for i in range(1,len(xTest)):
        tvec = xTest[i-1]+xTest[i]
        xTestAlter.append(tvec)
    clfr = gaussian_process.GaussianProcess(theta0=1e-2,
        thetaL=1e-4, thetaU=1e-1, corr=covar)
    clfr.fit(xTrainAlter,yTrain[1:])
    return clfr.predict(xTestAlter, eval_MSE=True)[0]

if __name__=="__main__":
    gaussianProcesses()