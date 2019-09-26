==================
LGM-Classification
==================

A python library for accurate classification of Points of Interest (POIs) into categories.

LGM-Classification is a python library implementing a full Machine Learning workflow for training classification algorithms on annotated POI datasets and producing models for the accurate classification of Points of Interest (POIs) into categories. LGM-Classification implements a series of training features, regarding the properties POIs and their relations with neighboring POIs. Further, it it encapsulates grid-search and cross-validation functionality, based on the [scikit](https://scikit-learn.org/) toolkit, assessing as series of classification models and parameterizations, in order to find the most fitting model for the data at hand.

The source code was tested using Python 3.6 and Scikit-Learn 0.21.2 on a Linux server.

Setup procedure
---------------

Download the latest version from the `GitHub repository <https://github.com/LinkGeoML/LGM-Classification.git>`_, change to
the main directory and run:

.. code:: bash

   pip install -r pip_requirements.txt

It should install all the required libraries automatically (*scikit-learn, numpy, pandas etc.*).
