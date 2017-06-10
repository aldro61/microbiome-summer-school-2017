# Microbiome Summer School 2017
## Introduction to machine learning

### Table of contents
1. [Introduction](#introduction)
    1. [Overview of our toolbox](#overview-of-our-toolbox)
    2. [Getting started](#getting-started)
    3. [Objectives](#objectives)
2. [The basics](#the-basics)
    1. [Type of learning problems](#type-of-learning-problems)
    2. [Typical experimental protocol](#typical-experimental-protocol)
    3. [Cross-validation](#cross-validation)
    4. [Assessing the accuracy of a model](#assessing-the-accuracy-of-a-model)
    5. [Interpretable vs black-box models](#interpretable-vs-black-box-models)
3. [Application: peptide protein binding affinity prediction](#application-peptide-protein-binding-affinity-prediction)
4. [Application: predicting antibiotic resistance](#application-predicting-antibiotic-resistance)
5. [Bonus: machine learning competition](#bonus-machine-learning-competition)
6. [Conclusion](#conclusion)


### Introduction

#### Overview of our toolbox

Present the tools that we will be using: Python, R, Sckit-Learn, Numpy

#### Getting started

Explain where to find the code and the data. How the practical session will be organized.

#### Objectives

Explain which competences we expect the participants to have gained following this tutorial.

___
### The basics

#### Type of learning problems

In supervised machine learning, data sets are collections of learning examples of the form *(x, y)*, where *x* is a **feature vector** and *y* is its corresonding **target value**. Features are observed variables that describe each example. The goal of a machine learning algorithm is to produce a **model** (*h*) that **accurately estimates** *y* given *x*, i.e., *h(x) &asymp; y*.

**Classification:** Each learning example is associated with a **qualitative** target value, which corresponds to a class (e.g., cancer, healthy). There can be two classes ([binary classification](https://en.wikipedia.org/wiki/Binary_classification)) or more ([multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)).

**Regression:** Each learning example is associated with a **quantitative** target value (e.g., survial time). The goal of the model is to estimate the correct output, given a learning example.

![Alt text](figures/figure.classification.vs.regression.png)


#### Typical experimental protocol

Show the two standard experimental protocols: train/validation/test and train(with k-fold cv)/test

#### Cross-validation

Give a bit more details about cross-validation and make them experiment with it a little bit.

#### Assessing the accuracy of a model

Show various metrics to measure the accuracy of a model

#### Interpretable vs black-box models

Compare various types of models. Use the figure where we see the decision boundary of many algorithms. Make them look at the coefficients of a linear model vs a simple decision tree model.

![Alt text](https://github.com/aldro61/pyscm/raw/master/examples/decision_boundary.png)

___
### Application: peptide protein binding affinity prediction

Use a data set from the work of Giguère et al. to show them a sequence-based regression problem. Use various algorithms: decision tree regression, kernel SVM and nearest neighbour. Benchmark their prediction accuracy using figures and make them notice the key differences between the algorithms. Explain why some work better than the others.


### Application: predicting antibiotic resistance

Use a data set from the PATRIC database to make them experiment with a really large scale classification problem. Make them train a few algorithms, including the SCM. Make them look at the SCM models and use BLAST to interprete the obtained model.


### Bonus: machine learning competition

Some participants will necessarily be faster than others. We could have a small machine learning competition with a leaderboard (use the one we use to train our rookies). 


### Conclusion

Wrap up and summarize what we have learned.
