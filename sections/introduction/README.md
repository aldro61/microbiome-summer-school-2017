<a href="../../#table-of-contents"><-- Back to table of contents</a>

# Introduction

## Overview of our toolbox

This tutorial is an introduction to machine learning in Python. We will rely heavily on the scikit-learn and numpy packages. Scikit-learn will be used for implementing machine learning protocols and learning algorithms, whereas numpy will be used to manipulate matrices of data.

<img src="https://www.python.org/static/img/python-logo@2x.png" height="50" />

<img src="http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" height="50" /> 

<img src="http://www.numpy.org/_static/numpy_logo.png" height="50" />
 
 
## Getting started

Clone the GitHub repository for the tutorial:
 
```bash
git clone https://github.com/aldro61/microbiome-summer-school-2017.git microbiome-ml-tutorial
```

Then, go to the exercise directory:

```bash
cd microbiome-ml-tutorial/exercise
```

Install the dependencies for the tutorial by running the following command:

```bash
sudo apt-get install virtualenv && \
virtualenv env && \
source ./env/bin/activate && \
pip install numpy --upgrade && \
pip install scipy --upgrade && \
pip install matplotlib --upgrade && \
pip install scikit-learn --upgrade && \
pip install seaborn --upgrade
```

## Objectives

After completing this tutorial, you should have acquired the following skills:
* Using Python to apply machine learning algorithms to data sets.
* Understanding the various types of problems (regression, classification)
* Perform machine learning experiments using correct protocols.
* Applying learning algorithms to biological data
