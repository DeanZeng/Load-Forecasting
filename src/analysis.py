#! /usr/bin/python

import visualizer
import statistics
import numpy as np
from datagen import constructData

'''
Methods to construct various statistical plots for the load time series
'''

# Plot the original load series
def plotOriginal():
    data = constructData()
    # Plot of aggregate electricity demand over the past 5 years
    section = data[1][0:len(data[1])-365]
    visualizer.yearlyPlot(section,
        2009,1,1,"Average Total Electricity Load : 2009-2013","Kilowatts")

# Plot the load series after detrending with least squares
# linear regression
def plotDetrended():
    data = constructData()
    # Plot of data after detrending with least squares regression
    indices = np.arange(len(data[1]))
    detrendY = statistics.detrend(indices,data[1])[0]
    visualizer.yearlyPlot(detrendY,
        2009,1,1,"Detrended Aggregate Electricity Demand","Residual Kilowatts")

# Plot the correlogram for the load series
# - plots autoregressive correlation coefficients against time lags
def plotCorrelogram():
    data = constructData()
    visualizer.autoCorrPlot(data[1][len(data[1])-730:len(data[1])-365],"Average Total Electricity Load Autocorrelations : 2013")

# Plot the lag plot for the load series
# - use to determine whether time series data is non-random
def plotLag():
    data = constructData()
    visualizer.lagPlot(data[1][0:len(data[1])-365],"Average Total Electricity Load Lag : 2009-2013")
        
# Plot the periodogram for the load series
# - plots frequency strength against frequencies over the spectrum
def plotPeriodogram():
    data = constructData()
    visualizer.periodogramPlot(data[1][len(data[1])-730:len(data[1])-365],
        "Periodogram of Average Total Electricity Load : 2013")

# Plot the original load series vs. the detrended load series
def plotOrigVsDetrend():
    data = constructData()
    # Original time series
    data1 = constructData()
    origY = data1[1][0:len(data[1])-365]
    # Detrended time series
    indices = np.arange(len(data[1])-365)
    detrendY = statistics.detrend(indices,data[1][0:len(data[1])-365])[0]

    visualizer.comparisonPlot(2009,1,1,origY,detrendY,
        "Original","Detrended",plotName="Aggregate Electric Load : Original & Detrended", yAxisName="Kilowatts")

if __name__=="__main__":
    plotCorrelogram()