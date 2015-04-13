#! /usr/bin/python

import visualizer
import math
import statistics
import numpy as np

from datagen import constructData
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.decomposition import PCA

# Neural Network load forecasting
def neuralNetwork():
  
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

  xTrain = [[math.log(y) for y in x] for x in xTrain]
  xTest = [[math.log(y) for y in x] for x in xTest]
  yTrain = [math.log(x) for x in yTrain]

  # Detrend the time series
  indices = np.arange(len(data[1]))
  trainIndices = indices[0:cutoff]
  testIndices = indices[cutoff:]
  detrended,slope,intercept = statistics.detrend(trainIndices,yTrain)
  yTrain = detrended

  dimensions = [6,10,12]
  neurons = [30,50,50]

  names = []
  for x in range(len(dimensions)):
    s = "d=" + str(dimensions[x]) + ",h=" + str(neurons[x])
    names.append(s)

  preds = []

  for x in range(len(dimensions)):

    # Perform dimensionality reduction on the feature vectors
    pca = PCA(n_components=dimensions[x])
    pca.fit(xTrain)
    xTrainRed = pca.transform(xTrain)
    xTestRed = pca.transform(xTest)

    pred = fit_predict(xTrainRed,yTrain,xTestRed,40,neurons[x])

    # Add the trend back into the predictions
    trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
    # Reverse the normalization
    trendedPred = [math.exp(x) for x in trendedPred]
    # Compute the NRMSE
    err = statistics.normRmse(yTest,trendedPred)
    
    # Append computed predictions to list for classifier predictions
    preds.append(trendedPred)

    print "The NRMSE for the neural network is " + str(err) + "..."

  preds.append(yTest)
  names.append("actual")

  visualizer.comparisonPlot(2014,1,1,preds,names,plotName="Neural Network Load Predictions vs. Actual", 
        yAxisName="Predicted Kilowatts")

# Constructs and fits a neural network with the given number of neurons
# to the training data for the specified number of epochs and returns a 
# vector of the predicted values for the given test data - assumes the target 
# is univariate (e.g. single valued output)
def fit_predict(xTrain,yTrain,xTest,epochs,neurons):

  # Check edge cases
  if (not len(xTrain) == len(yTrain) or len(xTrain) == 0 or 
    len(xTest) == 0 or epochs <= 0):
    return

  # Randomize the training data (probably not necessary but pybrain might
  # not shuffle the data itself, so perform as safety check)
  indices = np.arange(len(xTrain))
  np.random.shuffle(indices)

  trainSwapX = [xTrain[x] for x in indices]
  trainSwapY = [yTrain[x] for x in indices]

  supTrain = SupervisedDataSet(len(xTrain[0]),1)
  for x in range(len(trainSwapX)):
    supTrain.addSample(trainSwapX[x],trainSwapY[x])

  # Construct the feed-forward neural network

  n = FeedForwardNetwork()

  inLayer = LinearLayer(len(xTrain[0]))
  hiddenLayer1 = SigmoidLayer(neurons)
  outLayer = LinearLayer(1)

  n.addInputModule(inLayer)
  n.addModule(hiddenLayer1)
  n.addOutputModule(outLayer)

  in_to_hidden = FullConnection(inLayer, hiddenLayer1)
  hidden_to_out = FullConnection(hiddenLayer1, outLayer)
  
  n.addConnection(in_to_hidden)
  n.addConnection(hidden_to_out)

  n.sortModules() 

  # Train the neural network on the training partition, validating
  # the training progress on the validation partition

  trainer = BackpropTrainer(n,dataset=supTrain,momentum=0.1,learningrate=0.01
    ,verbose=False,weightdecay=0.01)
  
  trainer.trainUntilConvergence(dataset=supTrain,
    maxEpochs=epochs,validationProportion=0.30)

  outputs = []
  for x in xTest:
    outputs.append(n.activate(x))

  return outputs

if __name__ == "__main__":
  neuralNetwork()