#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
---------------------------------------------------------------------
Copyright 2011, 2012, 2013 Sébastien Giguère

This file is part of GSkernel

GSkernel is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GSkernel is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GSkernel.  If not, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------

This module will compute a square positive semi-define gram matrix or a
possibly rectangle matrix (for testing examples) using the GS kernel.

This module accept two file containing sequences, to obtain a square matrix
for training your algorithm you need to pass twice the training sequences.
To obtain a testing matrix first pass the training sequences then, the testing sequences.

USAGE:

This module takes exactly 8 arguments in the following order:

* Training sequences file path
* Testing sequences file path (can be the same as training)
* Amino acids property file path
* Float value for \sigma_p. Control the position uncertainty of sub-strings in the GS kernel.
* Float value for \sigma_c. Control the trade of between the amino-acids properties and the dirac delta.
* Integer value. Length of the sub-strings. Should not exceed sequences length.
* Boolean value (True/False). Normalize the gram matrix. We recommend to normalize.

Example:

python gs_kernel_slow.py ../examples/data/Zhou2010_cationic.dat ../examples/data/Zhou2010_cationic.dat ../amino_acids_matrix/AA.blosum50.dat 1.0 1.0 2 True train_matrix
'''

import numpy as np
from os import path
from sys import argv, exit
import time

def gs_kernel(str1, str2, psiDict, sigmaPos, sigmaAA, L):
    """
    Implementation of the GS (i.e. Generic String) kernel.
    
    The proposed kernel is designed for small bio-molecules (such as peptides) and pseudo-sequences
    of binding interfaces. The GS kernel is an elegant generalization of eight known kernels for local signals.
    Despite the richness of this new kernel, we provide a simple and efficient dynamic programming
    algorithm for its exact computation.
    
    str1, str2 -- Both sequence of amino acids to compare.
    sigmaPos -- Control the position uncertainty of the sub-strings of str1 and str2.
    sigmaAA -- Control the influence of amino acids properties (physico-chemical for example),
               can by any kind of properties.
    L -- Length of substrings
    """
    len_str1 = len(str1)
    len_str2 = len(str2)

    # A[i,j] contain the euclidean distance between
    # the i-th amino acid of str1 and the j-th amino acid of str2
    A = np.zeros((len_str1, len_str2))
    for i in xrange(len_str1):
        for j in xrange(len_str2):
            try:
                A[i,j] = psiDict[str1[i], str2[j]]
            except:
                # Default distance for unknown (and different) amino acid is 4, 0 otherwise
                if str1[i] != str2[j]:
                    A[i,j] = 4.0
    A = A / (-2.0 * (sigmaAA**2.0))
    A = np.exp(A)

    # Matrix B is used for dynamic programming,
    # for more detail, see the manuscript
    B = np.zeros((len_str1, len_str2))
    for i in xrange(len_str1):
        for j in xrange(len_str2):
            # tmp is used to exploit the recurrence of equation (10), see manuscript
            tmp = 1.0
            for l in xrange(L):
                if i+l < len_str1 and j+l < len_str2:
                    tmp *= A[i+l, j+l]
                    B[i,j] += tmp
    
    # Matrix P is used to store position uncertainty between positions
    # This matrix can be pre-computed and is independent of str1 and str2.
    # For simplicity we don't.
    P = np.zeros((len_str1, len_str2))
    for i in xrange(len_str1):
        for j in xrange(len_str2):
            P[i,j] = i-j
    
    P = np.square(P)
    P /= -2.0 * (sigmaPos ** 2.0)
    P = np.exp(P)

    return np.sum(P*B)


def compute_psi_dict(amino_acids, aa_descriptors):
    """
    This function pre-compute the square Euclidean distance
    between all amino acids descriptors and stock the distance
    in an hash table for easy and fast access during the
    GS kernel computation.

    amino_acids -- List of all amino acids in aa_descriptors
    
    aa_descriptors -- The i-th row of this matrix contain the
        descriptors of the i-th amino acid of amino_acids list.
    """
    
    # For every amino acids couple (a_1, a_2) psiDict is a hash table
    # that contain the squared Euclidean distance between the descriptors
    # of a_1 and a_2
    psiDict = {}
    
    # Fill the hash table psiDict
    for i in xrange(len(amino_acids)):
        for j in xrange(len(amino_acids)):
            c = aa_descriptors[i] - aa_descriptors[j]
            psiDict[amino_acids[i], amino_acids[j]] = np.dot(c,c)

    return psiDict


def load_AA_matrix(matrix_path):
    """
    Load the amino acids descriptors.
    Return the list of amino acids and a matrix where
    each row are the descriptors of an amino acid.
    
    matrix_path -- Path to the file containing the amino acids descriptors.
        See the amino_acid_matrix folder for the file format.
    """
    
    # Read the file
    f = open(path.expandvars(matrix_path), 'r')
    lines = f.readlines()
    f.close()

    amino_acids = []
    nb_descriptor = len(lines[0].split()) - 1
    aa_descriptors = np.zeros((len(lines), nb_descriptor))
    
    # Read descriptors
    for i in xrange(len(lines)):
        s = lines[i].split()
        aa_descriptors[i] = np.array([float(x) for x in s[1:]])
        amino_acids.append(s[0])

    # If nb_descriptor == 1, then all normalized aa_descriptors will be 1
    if nb_descriptor > 1:
        # Normalize each amino acid feature vector
        for i in xrange(len(aa_descriptors)):
            aa_descriptors[i] /= np.sqrt(np.dot(aa_descriptors[i],aa_descriptors[i]))

    return amino_acids, aa_descriptors

def gs_gram_matrix(X, Y, amino_acid_property_file, sigma_position = 1.0, sigma_amino_acid = 1.0, substring_length = 2, normalize_matrix = True):
    """
    Return the gram matrix K using the GG kernel such that K_i,j = k(x_i, y_j).
    If X == Y, M is a squared positive semi-definite matrix.
    When training call this function with X == Y, during the testing phase
    call this function with X containing the training examples and Y containing
    the testing examples.
    
    We recommend to normalize both the training and testing gram matrix.
    If the training matrix is normalized so should be the testing matrix.
    If the training matrix is un-normalized so should be the testing matrix.

    X -- List of examples, can be any kind of amino acid sequence
        (peptides, small protein, binding interface pseudo-sequence, ...)
        
    Y -- Second list of examples, can be the same as X for training or
        the testing examples.
        
    amino_acid_property_file -- Path to a file containing amino acid properties.
        See file example provided with this package for the simple file format.
    
    sigma_position -- Float value for \sigma_p. Control the position uncertainty of
        sub-strings in the GS kernel.
        Values in [0.0, 16.0] seem to empirically work well.
        
    sigma_amino_acid -- Float value for \sigma_c. Control the trade of between the
        amino-acids properties and the dirac delta.
        Values in [0.0, 16.0] seem to empirically work well.
    
    substring_length -- Length of the sub-strings. Should smaller or equal that the sequences in X or Y.
        Values in [1,6] seem to empirically work well.
        
    normalize_matrix -- Normalize the gram matrix. We recommend to normalize.
    """

    # X and Y should be np.array
    X = np.array(X)
    Y = np.array(Y)

    # Load amino acids descriptors
    (amino_acids, aa_descriptors) = load_AA_matrix(amino_acid_property_file)
    
    # For every amino acids couple (a_1, a_2) psi_dict is a hash table
    # that contain the squared Euclidean distance between the descriptors
    # of a_1 and a_2
    psi_dict = compute_psi_dict(amino_acids, aa_descriptors)
    
    # Declaration of the Gram matrix
    K = np.zeros((len(X), len(Y)))
    
    if X.shape == Y.shape and np.all(X == Y):
        
        # Fill the symmetric matrix
        for i in xrange(len(X)):
            K[i,i] = gs_kernel(X[i], X[i], psi_dict, sigma_position, sigma_amino_acid, substring_length)
            for j in xrange(i):
                K[i,j] = gs_kernel(X[i], X[j], psi_dict, sigma_position, sigma_amino_acid, substring_length)
                K[j,i] = K[i,j]
                
        if normalize_matrix:
            normX = np.sqrt(K.diagonal())
            K = ((K/normX).T/normX).T
    else:
        
        # Fill the non-symetric possibly rectangle matrix
        for i in xrange(len(X)):
            for j in xrange(len(Y)):
                K[i,j] = gs_kernel(X[i], Y[j], psi_dict, sigma_position, sigma_amino_acid, substring_length)
        
        if normalize_matrix:
            normX = np.sqrt([gs_kernel(x, x, psi_dict, sigma_position, sigma_amino_acid, substring_length) for x in X])
            normY = np.sqrt([gs_kernel(y, y, psi_dict, sigma_position, sigma_amino_acid, substring_length) for y in Y])
            K = ((K/normY).T/normX).T
            
    return K


if __name__ == '__main__':
    
    # Parsing of parameters
    try:
        X_file = argv[1]     # File containing one peptide per line
        Y_file = argv[2]     # Second file (can be the same as first) containing one peptide per line
        amino_acid_property_file = argv[3]  # Amino acid property file, see load_AA_matrix for file format
        sigma_position = float(argv[4])     # \sigma_p float value
        sigma_amino_acid = float(argv[5])   # \sigma_c float value
        substring_length = int(argv[6])     # Sub string length, integer value
        
        # Normalization of the gram matrix
        if argv[7].upper() == "TRUE":
            normalize_matrix = True
        elif argv[7].upper() == "FALSE":
            normalize_matrix = False
        else:
            raise ValueError
        
        output_file = argv[8] # Plain text output file name
    except:
        print __doc__
        exit()
        
    print "\n------- kernel parameters --------"
    print "X =", X_file
    print "Y =", Y_file
    print "Amino Acid property file =", amino_acid_property_file
    print "\sigma_p =", sigma_position
    print "\sigma_c =", sigma_amino_acid
    print "Substring length =", substring_length
    print "Normalize matrix =", normalize_matrix
    print "----------------------------------\n"
    
    print "Loading sequences ..."
    # Read sequences from the first file
    f = open(X_file)
    X = [l.split()[0] for l in f.readlines()]
    f.close()
    
    f = open(Y_file)
    Y = [l.split()[0] for l in f.readlines()]
    f.close()
    
    # Compute the Gram matrix
    print "Computing gram matrix using the GS kernel ..."
    
    t_1 = time.time() # profile the time
    K = gs_gram_matrix(X, Y, amino_acid_property_file, sigma_position, sigma_amino_acid, substring_length, normalize_matrix)
    t_2 = time.time()
    
    print "Gram matrix computation took", t_2-t_1,"seconds."
    
    # Writing matrix in output file
    print "Writing matrix in output file ..."
    np.savetxt(output_file, K)
    
    print "\nSuccess!" 
    
