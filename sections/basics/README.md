# The basics

## Type of learning problems

In supervised machine learning, data sets are collections of learning examples of the form *(x, y)*, where *x* is a **feature vector** and *y* is its corresonding **target value**. Features are observed variables that describe each example. The goal of a machine learning algorithm is to produce a **model** (*h*) that **accurately estimates** *y* given *x*, i.e., *h(x) &asymp; y*.

* **Classification:** Each learning example is associated with a **qualitative** target value, which corresponds to a class (e.g., cancer, healthy). There can be two classes ([binary classification](https://en.wikipedia.org/wiki/Binary_classification)) or more ([multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)).

* **Regression:** Each learning example is associated with a **quantitative** target value (e.g., survial time). The goal of the model is to estimate the correct output, given a feature vector.

![Alt text](../../figures/figure.classification.vs.regression.png)

![#1589F0](https://placehold.it/15/1589F0/000000?text=+) **Note:** There exists another type of supervised learning problem called [structured output prediction](https://en.wikipedia.org/wiki/Structured_prediction). This setting includes classification and regression, in addition to the prediction of complex structures, such as texts and images. However, this goes beyond the scope of this tutorial.


## Typical experimental protocol

### Training and testing sets

When performing a machine learning analysis on a data set, it is important to keep some data aside to estimate the accuracy the learned model.

* **Training set:** set of learning examples used to train the algorithm
* **Testing set:** set of learning examples used to estimate the accuracy of the model

<center>
	<img src="figures/train_test_split.png" height="175">
</center>

~~~python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
~~~

### Cross-validation

Some learning algorithms have **parameters** that must be **set by the user**. Such parameters are called **hyperparameters**. For example, when learning decision trees, the maximum depth of the tree is a hyperparameter.

Usually, we try many values for each hyperparameter and select the values that lead to the most accurate model.

![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Question:** Can we use the testing set to select the hyperparameter values that lead to the most accurate model?

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) **Answer:** No! Doing so would reveal information from the testing set to the learning algorithm.

Instead, we use a method called k-fold cross-validation:

Figure goes here.

~~~python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
param_grid = {'max_depth': [1, 5, 10, 50, 100]}
cv = GridSearchCV(DecisionTreeClassifier(), param_grid)
~~~

### Assessing the accuracy of a model

Show various metrics to measure the accuracy of a model

## Interpretable vs black-box models

Compare various types of models. Use the figure where we see the decision boundary of many algorithms. Make them look at the coefficients of a linear model vs a simple decision tree model.

![Alt text](https://github.com/aldro61/pyscm/raw/master/examples/decision_boundary.png)