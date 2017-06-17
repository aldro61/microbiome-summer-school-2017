__author__ = 'Alexandre'

import cPickle
import numpy as np
import os
import random
from math import exp
from time import time, sleep
from GSkernel import load_AA_matrix, GS_kernel, compute_psi_dict
from GSkernel_fast import GS_gram_matrix_fast


def GS_kernel_naive(str1, str2, sigmaPos, sigmaAA, L, amino_acids, aa_descriptors):
    k_value = 0.0
    denom_p = -2.0 * (sigmaPos ** 2)
    denom_c = -2.0 * (sigmaAA ** 2)

    aa_indexes = {}
    for i in xrange(len(amino_acids)):
        aa_indexes[amino_acids[i]] = i

    for l in xrange(L):
        l += 1
        for i in xrange(len(str1) - l + 1):
            #Compute phi^l(x_i+1, .., x_i+l)
            s1 = str1[i:i + l]
            phi_l_str1 = np.array([])
            for aa in s1:
                phi_aa = aa_descriptors[aa_indexes[aa]]
                phi_l_str1 = np.append(phi_l_str1, phi_aa)

            for j in xrange(len(str2) - l + 1):
                dist_p = i - j

                # Compute phi^l(x_j+1, .., x_j+l)
                s2 = str2[j:j + l]
                phi_l_str2 = np.array([])
                for aa in s2:
                    phi_aa = aa_descriptors[aa_indexes[aa]]
                    phi_l_str2 = np.append(phi_l_str2, phi_aa)

                dist_c = np.linalg.norm(phi_l_str1 - phi_l_str2)

                k_value += exp((dist_p ** 2) / denom_p) * exp((dist_c ** 2) / denom_c)
    return k_value


def GS_kernel_precomp_P(str1, str2, psiDict, sigmaPos, sigmaAA, L, P):
    len_str1 = len(str1)
    len_str2 = len(str2)

    A = np.zeros((len_str1, len_str2))
    for i in xrange(len_str1):
        for j in xrange(len_str2):
            try:
                A[i, j] = psiDict[str1[i], str2[j]]
            except:
                if str1[i] != str2[j]:
                    A[i, j] = 4.0
    A /= -2.0 * (sigmaAA ** 2.0)
    A = np.exp(A)

    B = np.zeros((len_str1, len_str2))
    for i in xrange(len_str1):
        for j in xrange(len_str2):
            tmp = 1.0
            for l in xrange(L):
                if i + l < len_str1 and j + l < len_str2:
                    tmp *= A[i + l, j + l]
                    B[i, j] += tmp

    return np.sum(P * B)


def precompute_P(len_x_1, len_x_2, sigma_pos):
    P = np.zeros((len_x_1, len_x_2))

    for i in xrange(len_x_1):
        for j in xrange(len_x_2):
            P[i, j] = i - j

    P = np.square(P)
    P /= -2.0 * (sigma_pos ** 2.0)
    P = np.exp(P)

    return P


def GS_gram_matrix(kernel_func, X, amino_acid_property_file, sigma_position=1.0, sigma_amino_acid=1.0,
                   substring_length=2):
    if kernel_func == 'GS_kernel_naive' or kernel_func == 'GS_kernel_precomp_P':
        # Load amino acids descriptors
        (amino_acids, aa_descriptors) = load_AA_matrix(amino_acid_property_file)

        # For every amino acids couple (a_1, a_2) psi_dict is a hash table
        # that contain the squared Euclidean distance between the descriptors
        # of a_1 and a_2
        psi_dict = compute_psi_dict(amino_acids, aa_descriptors)

        # Declaration of the Gram matrix
        K = np.zeros((len(X), len(X)))
        maxLen = max([len(s) for s in X])
        P = precompute_P(maxLen, maxLen, sigma_position)

        if kernel_func == 'GS_kernel_naive':
            # Fill the symmetric matrix
            for i in xrange(len(X)):
                K[i, i] = GS_kernel_naive(X[i],
                    X[i],
                    sigma_position,
                    sigma_amino_acid,
                    substring_length,
                    amino_acids,
                    aa_descriptors)
                for j in xrange(i):
                    K[i, j] = GS_kernel_naive(X[i],
                        X[j],
                        sigma_position,
                        sigma_amino_acid,
                        substring_length,
                        amino_acids,
                        aa_descriptors)
                    K[j, i] = K[i, j]

        elif kernel_func == 'GS_kernel_precomp_P':
            for i in xrange(len(X)):
                K[i, i] = GS_kernel_precomp_P(X[i],
                    X[i],
                    psi_dict,
                    sigma_position,
                    sigma_amino_acid,
                    substring_length,
                    P)
                for j in xrange(i):
                    K[i, j] = GS_kernel_precomp_P(X[i],
                        X[j],
                        psi_dict,
                        sigma_position,
                        sigma_amino_acid,
                        substring_length,
                        P)
                    K[j, i] = K[i, j]

    elif kernel_func == 'GS_kernel_fast':
        K = GS_gram_matrix_fast(X=X,
            Y=X,
            amino_acid_property_file=amino_acid_property_file,
            sigma_position=sigma_position,
            sigma_amino_acid=sigma_amino_acid,
            substring_length=substring_length,
            approximate=False,
            normalize_matrix=False)

    elif kernel_func == 'GS_kernel_fast_approx':
        K = GS_gram_matrix_fast(X=X,
            Y=X,
            amino_acid_property_file=amino_acid_property_file,
            sigma_position=sigma_position,
            sigma_amino_acid=sigma_amino_acid,
            substring_length=substring_length,
            approximate=True,
            normalize_matrix=False)

    return K


def generate_peptides(amino_acids, peptide_length, n_peptides):
    return [''.join(random.choice(amino_acids) for x in range(peptide_length)) for i in xrange(n_peptides)]


def pickle(filename, data):
    f = open(filename, 'wb')
    cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


def benchmark1():
    n_peptides = 50
    peptide_length = 15
    sigma_pos = 0.5
    sigma_aa = 0.5
    L_range = range(1, 16)
    n_run_average = 5
    X = generate_peptides('ARNDCQEGHILKMFPSTWYVBZX*', peptide_length=peptide_length, n_peptides=n_peptides)

    results = {}
    kernel_funcs = ['GS_kernel_naive', 'GS_kernel_precomp_P']

    for kernel in kernel_funcs:
        results[kernel] = []

    for L in L_range:
        print 'L = ', L
        for kernel in kernel_funcs:
            runtimes = []
            for i in xrange(n_run_average):
                t = time()
                GS_gram_matrix(kernel, X, '../amino_acids_matrix/AA.blosum50.dat', sigma_position=sigma_pos,
                    sigma_amino_acid=sigma_aa, substring_length=L)
                elapsed = time() - t
                runtimes.append(elapsed)
            result = np.median(runtimes)
            results[kernel].append(result)
            print kernel, ' completed in ', result, ' seconds (average over', n_run_average, 'runs).'
            sleep(0.1) # allow to switch CPU
            #Checkpoint! Save intermediate results
        pickle('benchmark1_results_L=' + str(L) + '.pkl', results)
    print

    print 'Saving results...'
    pickle('benchmark1_results.pkl', results)
    for f in os.listdir('.'):
        import re

        if re.search('benchmark1_results_L', f):
            os.remove(f)
    print 'Done.'
    print

    print 'Matlab:'
    print '-' * 50
    print 'Lrange =', L_range, ';'
    for kernel in kernel_funcs:
        print kernel, '=', results[kernel], ';'
    print 'clf;'
    print 'hold all;'
    for kernel in kernel_funcs:
        print 'plot(Lrange,', kernel, ');'

    print "legend('" + "','".join([x for x in kernel_funcs]) + "');"


def benchmark2():
    n_peptides = 1000
    peptide_length = 100
    sigma_pos_range = np.arange(0.1, 30.0, 0.5)
    sigma_aa = 0.5
    L = 2
    n_run_average = 15

    results = {}
    kernel_funcs = ['GS_kernel_fast', 'GS_kernel_fast_approx']

    for kernel in kernel_funcs:
        results[kernel] = []

    for sigma_pos in sigma_pos_range:
        print 'sigma_pos = ', sigma_pos
        for kernel in kernel_funcs:
            runtimes = []
            for i in xrange(n_run_average):
                X = generate_peptides('ARNDCQEGHILKMFPSTWYVBZX*', peptide_length=peptide_length, n_peptides=n_peptides)
                t = time()
                GS_gram_matrix(kernel, X, '../amino_acids_matrix/AA.blosum50.dat', sigma_position=sigma_pos,
                    sigma_amino_acid=sigma_aa, substring_length=L)
                elapsed = time() - t
                runtimes.append(elapsed)
            result = np.median(runtimes)
            results[kernel].append(result)
            print kernel, ' completed in ', result, ' seconds (average over', n_run_average, 'runs).'

            #Checkpoint! Save intermediate results
        pickle('benchmark2_results_sigma_pos=' + str(sigma_pos) + '.pkl', results)
    print

    print 'Saving results...'
    pickle('benchmark2_results.pkl', results)
    for f in os.listdir('.'):
        import re

        if re.search('benchmark2_results_sigma_pos', f):
            os.remove(f)
    print 'Done.'
    print

    print 'Matlab:'
    print '-' * 50
    print 'sigma_pos_range =', list(sigma_pos_range), ';'
    for kernel in kernel_funcs:
        print kernel, '=', results[kernel], ';'
    print 'clf;'
    print 'hold all;'
    for kernel in kernel_funcs:
        print 'plot(sigma_pos_range, ', kernel, ');'

    print "legend('" + "','".join([x for x in kernel_funcs]) + "');"

if __name__ == '__main__':
    benchmark1()
    #benchmark2()