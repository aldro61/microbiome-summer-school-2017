"""
Train a Support Vector Machine based on the Ray Surveyor similarity matrix

Note: assumes that the Kover dataset is present at ./kover-example/example.kover

"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns; sns.set_palette("hls", 4)

from kover.dataset import KoverDataset
from sklearn.svm import SVC


data_dir = "./data/antibiotics/"
kover_example_path = "./kover-example/"
random_state = np.random.RandomState(42)

print "Loading the data..."
# Load the kover dataset
kover_dataset = KoverDataset(os.path.join(kover_example_path, "example.kover"))
kover_genome_ids = kover_dataset.genome_identifiers[...]

# Load the linear kernel matrix, i.e., the Ray Surveyor similarity matrix
similarity_data = pd.read_table(os.path.join(data_dir, "mtuberculosis_rifampicin_similarity.tsv"), index_col=0)
similarity_data = similarity_data[kover_genome_ids].loc[kover_genome_ids]  # reorder to match kover dataset
assert np.all(similarity_data.columns.values == similarity_data.index.values)  # make sure it's symmetric
K = similarity_data.values

# Get the labels
labels = kover_dataset.phenotype.metadata[...]

# Training and testing set split
print "Splitting the data into training and testing sets..."
kover_train_example_idx = kover_dataset.get_split("example_split").train_genome_idx[...]
kover_test_example_idx = kover_dataset.get_split("example_split").test_genome_idx[...]
K_train = K[kover_train_example_idx][:, kover_train_example_idx]
K_test = K[kover_test_example_idx][:, kover_train_example_idx]
y_train = labels[kover_train_example_idx]
y_test = labels[kover_test_example_idx]

# Assign each example to a cross-validation fold
print "Assigning each example to a cross-validation fold..."
n_folds = 5
cv_folds = np.arange(K_train.shape[0]) % n_folds
random_state.shuffle(cv_folds)

c_values = np.logspace(-7, 7, 20)
best_c = None
best_c_score = -np.infty
print "{0:d}-fold ross-validation for {1:d} C values...".format(n_folds, len(c_values))
for c in c_values:
    fold_scores = []
    for fold in np.unique(cv_folds):
        fold_K_train = K_train[cv_folds != fold][:, cv_folds != fold]
        fold_y_train = y_train[cv_folds != fold]
        fold_K_test = K_train[cv_folds == fold][:, cv_folds != fold]
        fold_y_test = y_train[cv_folds == fold]

        "Fitting with C={0:.4f}"
        estimator = SVC(C=c, kernel="precomputed").fit(fold_K_train, fold_y_train)
        fold_scores.append(estimator.score(fold_K_test, fold_y_test))
    cv_score = np.mean(fold_scores)
    print "... cv score: {0:.4f}".format(cv_score)
    if cv_score > best_c_score:
        best_c = c
        best_c_score = cv_score
print "The best C value is C = {0:.4f} with cv score = {1:.4f}".format(best_c, best_c_score)

print "Re-training with best hyperparameter values..."
estimator = SVC(C=best_c, kernel="precomputed").fit(K_train, y_train)
svm_train_score = estimator.score(K_train, y_train)
svm_test_score = estimator.score(K_test, y_test)

print "Loading Kover results..."
kover_results = json.load(open(os.path.join(kover_example_path, "results.json"), "r"))
kover_train_score = 1.0 - kover_results["metrics"]["train"]["risk"][0]
kover_test_score = 1.0 - kover_results["metrics"]["test"]["risk"][0]

width = 0.3
plt.bar([0], [kover_train_score], width, color="red", label="Kover (SCM)")
plt.bar([0 + width], [svm_train_score], width, color="blue", label="SVM")
plt.bar([1], [kover_test_score], width, color="red")
plt.bar([1 + width], [svm_test_score], width, color="blue")
plt.xticks([0.15, 1.18], ["Training set", "Testing set"])
plt.xlabel("")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for predicting rifampicin resistance in M. tuberculosis")
plt.show()