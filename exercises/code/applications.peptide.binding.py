#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""

TODO:
    * Code cleanup
    * Memoize kernel matrices to speed up CV (over C values)

"""
import h5py as h
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns; sns.set_palette("hls", 4)
import warnings

from gs_kernel.gs_kernel import gs_gram_matrix
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR as SupportVectorRegression

data_path = "./data/peptide/"
KERNEL_MATRIX_CACHE = os.path.join(data_path, "gs_kernel_matrix_cache.h5")
PHYSICOCHEMICAL_PROPERTY_FILE = os.path.join(data_path, "amino_acids_matrix/AA.blosum50.dat")


def sequences_to_vec(sequences, length=3):
    d = {}

    # Find all unique k-mers
    for s in sequences:
        for i in range(len(s) - (length - 1)):
            d[s[i : i + length]] = True
            assert len(s[i : i + length]) == length

    # Construct a count vector for each sequence
    seq_vec = np.zeros((len(sequences), len(d)), dtype=np.float)
    for i, s in enumerate(sequences):
        for j, kmer in enumerate(d):
            seq_vec[i, j] = s.count(kmer)

    return seq_vec


def cross_validation_gs_kernel(seq, is_train, folds):
    # Cross-validation
    # Repeat for each combination of candidate hyperparameter values
    sigma_position_values = np.hstack((1e-9, np.logspace(0, 1, 5)))
    sigma_amino_acid_values = np.hstack((1e-9, np.logspace(0, 1, 5)))
    substring_length = 9
    C_values = [0.01, 0.1, 1.0, 10., 100.]

    best_result_aa_pos = {"score": -np.infty}
    best_result_aa_only = {"score": -np.infty}
    best_result_pos_only = {"score": -np.infty}

    with h.File(KERNEL_MATRIX_CACHE, "w") as kernel_cache:
        for sigma_pos, sigma_aa, C in product(sigma_position_values, sigma_amino_acid_values, C_values):

            print "Parameters: sigma pos: {0:.4f}   sigma amino acid: {1:.4f}   C: {2:.4f}".format(sigma_pos, sigma_aa, C)
            kernel_matrix_id = "{0:.4f}{1:.4f}{2:d}".format(sigma_pos, sigma_aa, substring_length)
            if kernel_matrix_id in kernel_cache:
                K = kernel_cache[kernel_matrix_id][...]
            else:
                K = gs_gram_matrix(seq, seq,
                                   amino_acid_property_file=PHYSICOCHEMICAL_PROPERTY_FILE,
                                   sigma_position=sigma_pos,
                                   sigma_amino_acid=sigma_aa,
                                   substring_length=substring_length)
                kernel_cache.create_dataset(kernel_matrix_id, data=K)

            K_train = K[is_train][:, is_train]
            K_test = K[~is_train][:, is_train]
            y_train = labels[is_train]
            y_test = labels[~is_train]

            # Cross-validation
            fold_scores = []
            for fold in np.unique(folds):
                print "...Fold {0:d}".format(fold + 1)
                fold_K_train = K_train[folds != fold][:, folds != fold]
                fold_K_test = K_train[folds == fold][:, folds != fold]
                fold_y_train = y_train[folds != fold]
                fold_y_test = y_train[folds == fold]
                fold_estimator = SupportVectorRegression(kernel="precomputed", C=C)
                fold_estimator.fit(fold_K_train.copy(), fold_y_train)
                fold_scores.append(fold_estimator.score(fold_K_test, fold_y_test))
            cv_score = np.mean(fold_scores)
            print "...... cv score:", cv_score

            if cv_score > best_result_aa_pos["score"]:
                best_result_aa_pos["score"] = cv_score
                best_result_aa_pos["K"] = dict(train=K_train, test=K_test, full=K)
                best_result_aa_pos["estimator"] = SupportVectorRegression(kernel="precomputed", C=C).fit(K_train, y_train)
                best_result_aa_pos["hps"] = dict(sigma_position=sigma_pos, sigma_amino_acid=sigma_aa, C=C)

            if np.isclose(sigma_pos, 1e-9) and cv_score > best_result_aa_only["score"]:
                best_result_aa_only["score"] = cv_score
                best_result_aa_only["K"] = dict(train=K_train, test=K_test, full=K)
                best_result_aa_only["estimator"] = SupportVectorRegression(kernel="precomputed", C=C).fit(K_train, y_train)
                best_result_aa_only["hps"] = dict(sigma_position=sigma_pos, sigma_amino_acid=sigma_aa, C=C)

            if np.isclose(sigma_aa, 1e-9) and cv_score > best_result_pos_only["score"]:
                best_result_pos_only["score"] = cv_score
                best_result_pos_only["K"] = dict(train=K_train, test=K_test, full=K)
                best_result_pos_only["estimator"] = SupportVectorRegression(kernel="precomputed", C=C).fit(K_train, y_train)
                best_result_pos_only["hps"] = dict(sigma_position=sigma_pos, sigma_amino_acid=sigma_aa, C=C)

            print
            print

    return best_result_aa_pos, best_result_aa_only, best_result_pos_only


def cross_validation_spectrum_kernel(seq, is_train, folds):
    # Cross-validation
    # Repeat for each combination of candidate hyperparameter values
    substring_length = 3
    C_values = [0.01, 0.1, 1.0, 10., 100.]

    best_result = {"score": -np.infty}

    seq_vec = sequences_to_vec(seq, substring_length)
    K = np.dot(seq_vec, seq_vec.T)
    K_train = K[is_train][:, is_train]
    K_test = K[~is_train][:, is_train]
    y_train = labels[is_train]
    y_test = labels[~is_train]

    for C in C_values:

        print "Parameters: C: {0:.4f}".format(C)

        # Cross-validation
        fold_scores = []
        for fold in np.unique(folds):
            print "...Fold {0:d}".format(fold + 1)
            fold_K_train = K_train[folds != fold][:, folds != fold]
            fold_K_test = K_train[folds == fold][:, folds != fold]
            fold_y_train = y_train[folds != fold]
            fold_y_test = y_train[folds == fold]
            fold_estimator = SupportVectorRegression(kernel="precomputed", C=C)
            fold_estimator.fit(fold_K_train.copy(), fold_y_train)
            fold_scores.append(fold_estimator.score(fold_K_test, fold_y_test))
        cv_score = np.mean(fold_scores)
        print "...... cv score:", cv_score

        if cv_score > best_result["score"]:
            best_result["score"] = cv_score
            best_result["K"] = dict(train=K_train, test=K_test, full=K)
            best_result["estimator"] = SupportVectorRegression(kernel="precomputed", C=C).fit(K_train, y_train)
            best_result["hps"] = dict(C=C)

        print
        print
    return best_result


for ds in os.listdir(data_path):
    print ds
    if ".dat" not in ds:
        continue

    if "DRB1_0701" not in ds:
        continue

    random_state = np.random.RandomState(42)
    dataset_path = os.path.join(data_path, ds)
    seq, labels = zip(*[(l.strip().split()[1], l.strip().split()[2])for l in open(dataset_path, "r")])
    labels = np.array(labels, dtype=np.float)

    # Split the data set into a training (80% of the data) and testing set (20% of the data)
    is_train = random_state.binomial(1, 0.8, len(labels)).astype(np.bool)
    y_train = labels[is_train]
    y_test = labels[~is_train]

    # Assign each training example to a cross-validation fold
    n_folds = 5
    folds = np.arange(is_train.sum()) % n_folds
    random_state.shuffle(folds)

    best_result_aa_pos, best_result_aa_only, best_result_pos_only = cross_validation_gs_kernel(seq, is_train, folds)
    best_result_spectrum = cross_validation_spectrum_kernel(seq, is_train, folds)

    # Figure 1: GS kernel matrix with the selected hyperparameters
    plt.clf()
    cm = sns.clustermap(best_result_aa_pos["K"]["full"])
    cm.ax_heatmap.tick_params(labelbottom="off", labelright="off")
    cm.ax_col_dendrogram.set_title("Generic String Kernel Matrix for {0:d} peptides".format(len(labels)))
    plt.savefig("gs_kernel_low_res.png", dpi=100, bbox_inches="tight")
    plt.savefig("gs_kernel_high_res.png", dpi=400, bbox_inches="tight")
    plt.show()

    # Figure 2: Comparison of the predictive performance of GS kernel variants
    plt.clf()
    plt.clf()
    width = 0.5
    plt.bar([1], [best_result_aa_pos["estimator"].score(best_result_aa_pos["K"]["test"], y_test)], width, label="GS (Alignment + Physicochemical)")
    plt.bar([1 + width], [best_result_aa_only["estimator"].score(best_result_aa_only["K"]["test"], y_test)], width, label="GS (Physicochemical)")
    plt.bar([1 + 2 * width], [best_result_pos_only["estimator"].score(best_result_pos_only["K"]["test"], y_test)], width, label="GS (Alignment)")
    plt.xlabel("Method")
    plt.ylabel("Coefficient of determination ($r^2$)")
    plt.gca().tick_params(labelbottom='off')
    plt.legend()
    plt.legend()
    plt.savefig("gs_variants_low_res.png", dpi=100, bbox_inches="tight")
    plt.savefig("gs_variants_high_res.png", dpi=400, bbox_inches="tight")
    plt.show()

    # Figure 3: Comparison of the GS kernel and the Spectrum kernel
    plt.clf()
    plt.scatter(y_test, best_result_aa_pos["estimator"].predict(best_result_aa_pos["K"]["test"]),
                label="GS (Alignment + Physicochemical) [{0:.3f}]".format(
                    best_result_aa_pos["estimator"].score(best_result_aa_pos["K"]["test"], y_test)))
    plt.scatter(y_test, best_result_spectrum["estimator"].predict(best_result_spectrum["K"]["test"]),
                label="Spectrum [{0:.3f}]".format(
                    best_result_spectrum["estimator"].score(best_result_spectrum["K"]["test"], y_test)))
    plt.plot([0, y_test.max()], [0, y_test.max()], color="black")
    plt.xlabel("True binding affinity")
    plt.ylabel("Predicted binding affinity")
    plt.legend()
    plt.savefig("gs_vs_spectrum_low_res.png", dpi=100, bbox_inches="tight")
    plt.savefig("gs_vs_spectrum_high_res.png", dpi=400, bbox_inches="tight")
    plt.show()
