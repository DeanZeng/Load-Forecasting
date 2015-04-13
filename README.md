# Load-Forecasting

Using the freely available load time series data from Elia, this projects  aims to develop robust and accurate methods for forecasting average total load on the Elia electric grid one day ahead of time. There are four different machine learning based algorithms that have been developed, each of which can be ran by running the respective script in the src folder. They are as follows:

* Gaussian Process Regression - gpr.py
* Support Vector Regression - svr.py
* Weighted Clustering - clustering.py
* Sigmoidal Neural Network - neural.py

The data required by the scripts is stored in the data folder within src. The
script analysis.py provides functions to visualize various aspects of the Elia load time series. All simulations display resulting predictions using
methods from visualizer.py. For more details about the Elia dataset,
the development of the algorithms and the forecasting results, view the writeup in the writeup folder.

## Credit:

* Credit is given to Elia for providing the electricity load dataset.
* Source : http://www.elia.be/en/grid-data

## License:

* This project is licensed under the MIT open source license. Please see the LICENSE file for details.