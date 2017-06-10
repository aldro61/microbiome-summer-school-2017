"""
Shows example regression and classification problems

"""
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns; sns.set_style("white")


from sklearn.datasets import make_blobs, make_regression
from sklearn.svm import LinearSVC, LinearSVR


def make_classification_example(axis):
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2.7, random_state=random_state)

    axis.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Disease")
    axis.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Healthy")

    clf = LinearSVC().fit(X, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the line, the points, and the nearest vectors to the plane
    axis.plot(xx, yy, 'k-', color="black")

    ax1.tick_params(labelbottom='off', labelleft='off')
    ax1.set_xlabel("Gene 1")
    ax1.set_ylabel("Gene 2")
    ax1.legend()


def make_regression_example(axis):
    X, y = make_regression(n_samples=100, n_features=1, noise=30.0, random_state=random_state)

    axis.scatter(X[:, 0], y, color="blue", s=10, label="Patients")

    clf = LinearSVR().fit(X, y)
    axis.plot(X[:, 0], clf.predict(X), color="black")

    ax2.tick_params(labelbottom='off', labelleft='off')
    ax2.set_xlabel("Gene 1")
    ax2.set_ylabel("Survival time")
    ax2.legend()


random_state = np.random.RandomState(42)

f, (ax1, ax2) = plt.subplots(ncols=2)
f.set_size_inches(7, 3)

ax1.set_title("Classification")
make_classification_example(ax1)

ax2.set_title("Regression")
make_regression_example(ax2)

plt.savefig("figure.classification.vs.regression.png", bbox_inches="tight", dpi=300)

