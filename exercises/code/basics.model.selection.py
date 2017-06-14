import numpy as np

from itertools import product
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer, make_classification, make_circles, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

random_state = np.random.RandomState(3)

X, y = load_breast_cancer(return_X_y=True)

X, y = shuffle(X, y, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)

gamma_values = np.logspace(-8, 0, 10)
c_values = np.logspace(-6, 6, 10)
hp_combinations = list(product(gamma_values, c_values))

data_train = np.zeros((len(gamma_values), len(c_values)))
data_cv = np.zeros((len(gamma_values), len(c_values)))
data_test = np.zeros((len(gamma_values), len(c_values)))
for i in range(len(gamma_values)):
    for j in range(len(c_values)):
        print "Fitting with gamma={0:.8f} and C={1:.8f}".format(gamma_values[i], c_values[j])

        estimator = SVC(kernel="rbf", gamma=gamma_values[i], C=c_values[j])
        estimator.fit(X_train, y_train)

        data_train[i, j] = estimator.score(X_train, y_train)
        data_cv[i, j] = np.mean(cross_val_score(estimator, X_train, y_train, cv=5, n_jobs=-1))
        data_test[i, j] = estimator.score(X_test, y_test)

import matplotlib.pyplot as plt
import seaborn as sns
f, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
sns.heatmap(data_train, ax=ax1, cbar=False, annot=True, annot_kws=dict(fontsize=6))
sns.heatmap(data_cv, ax=ax2, cbar=False, annot=True, annot_kws=dict(fontsize=6))
sns.heatmap(data_test, ax=ax3, cbar=False, annot=True, annot_kws=dict(fontsize=6))
plt.suptitle("Accuracy of the learned model for various hyperparameter combinations")
ax1.set_title("Training set")
ax2.set_title("Cross-validation")
ax3.set_title("Testing set")

ax1.set_xlabel("RBF kernel gamma")
ax1.set_ylabel("SVM C")
ax2.set_xlabel("RBF Kernel Gamma")
ax2.set_ylabel("SVM C")
ax3.set_xlabel("RBF Kernel Gamma")
ax3.set_ylabel("SVM C")

ax1.set(adjustable='box-forced', aspect='equal')
ax2.set(adjustable='box-forced', aspect='equal')
ax3.set(adjustable='box-forced', aspect='equal')

f.set_size_inches(10, 4)
plt.tight_layout()
plt.show()