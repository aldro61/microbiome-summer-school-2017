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

This module takes exactly 9 arguments in the following order:

* Training sequences file path
* Testing sequences file path (can be the same as training)
* Amino acids property file path
* Float value for \sigma_p. Control the position uncertainty of sub-strings in the GS kernel.
* Float value for \sigma_c. Control the trade of between the amino-acids properties and the dirac delta.
* Integer value. Length of the sub-strings. Should not exceed sequences length.
* Boolean value (True/False). Normalize the gram matrix. We recommend to normalize.
* Boolean value (True/False). Approximate kernel value. Approximate _ONLY_ if you know what you are doing.

Example:

python gs_kernel.py ../examples/data/Zhou2010_cationic.dat ../examples/data/Zhou2010_cationic.dat ../amino_acids_matrix/AA.blosum50.dat 1.0 1.0 2 True False train_matrix
'''


from gs_kernel_cython import  GS_gram_matrix_cython, GS_diagonal_cython
from gs_kernel_slow import load_AA_matrix, compute_psi_dict
import numpy as np
from sys import argv, exit
import time
from math import ceil

def amino_acid_to_int( sequence_list, max_length):
    """
    This function convert amino acids string
    into int array for faster computation.
    """
    
    sequence_int = np.zeros((len(sequence_list), max_length), dtype=np.int8) - 1
    
    for i in xrange(len(sequence_list)):
        for j in xrange(len(sequence_list[i])):
            sequence_int[i,j] = ord(sequence_list[i][j])
    
    return sequence_int


def compute_P(max_string_length, sigma_position):
    """
    P is a matrix that contains all possible position
    uncertainty values. This function pre-compute all
    possible values since those values are independant of
    the amino acids sequence. 
    """
    
    P = np.zeros((max_string_length, max_string_length))
    
    for i in xrange(max_string_length):
        for j in xrange(max_string_length):
            P[i,j] = i-j
    
    P = np.square(P)
    P /= -2.0 * (sigma_position ** 2.0)
    P = np.exp(P)
    
    return P

def psi_dict_to_matrix(psi_dict, sigma_amino_acid):
    """
    It is possible to get a perfect hash using a 128x128 matrix containing
    the squared Euclidean distance between all possible amino acids.
    
    The hashing function is simply the ASCII value of the amino acid.
    """
    
    N = 128
    psi_matrix = np.zeros((N,N))+4
    idx = np.arange(N)
    psi_matrix[idx,idx] = 0 # set diagonal to zero
    
    for key, val in psi_dict.items():
        i = ord(key[0])
        j = ord(key[1])
        psi_matrix[i,j] = val
        
    return np.exp( -psi_matrix/(2.0 * (sigma_amino_acid**2))) 

def gs_gram_matrix(X, Y, amino_acid_property_file, sigma_position = 1.0, sigma_amino_acid = 1.0, substring_length = 2, approximate = False, normalize_matrix = True):
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
    
    l -- Length of the sub-strings. Should smaller or equal that the sequences in X or Y.
        Values in [1,6] seem to empirically work well.
        
    normalize_matrix -- Normalize the gram matrix. We recommend to normalize.
    """
    
    # Load amino acids descriptors
    (amino_acids, aa_descriptors) = load_AA_matrix(amino_acid_property_file)
    
    # For every amino acids couple (a_1, a_2) psi_dict is a hash table
    # that contain the squared Euclidean distance between the descriptors
    # of a_1 and a_2
    psi_dict = compute_psi_dict(amino_acids, aa_descriptors)
    
    # Find the maximum length of string in X and Y
    max_string_length = max(max([len(x) for x in X]), max([len(y) for y in Y]))
    
    # It is possible to get a perfect hash using a 128x128 matrix containing
    # the squared Euclidean distance between all possible amino acids.
    psi_matrix = psi_dict_to_matrix(psi_dict, sigma_amino_acid)
    
    # Matrix P is used to store position uncertainty between positions
    # This matrix is independent of the amino acids sequence.
    P = compute_P(max_string_length, sigma_position)
    
    X_length = np.array([len(x) for x in X], dtype=np.int64)
    Y_length = np.array([len(y) for y in Y], dtype=np.int64)
    
    
    X_int = amino_acid_to_int(X, max_string_length)
    Y_int = amino_acid_to_int(Y, max_string_length)
    
    symetric_matrix = X_int.shape == Y_int.shape and np.all(X_int == Y_int)
    delta = int(ceil(3.0 * sigma_position))
    
    K = GS_gram_matrix_cython( X_int, X_length, Y_int, Y_length, psi_matrix, P, substring_length, delta, symetric_matrix, approximate)
    
    # Normalization of the matrix
    if normalize_matrix:
        if symetric_matrix:
            norm_X = np.sqrt(np.diagonal(K))
            norm_Y = norm_X
        else:
            norm_X = np.sqrt(GS_diagonal_cython( X_int, X_length, psi_matrix, P, substring_length, delta, approximate))
            norm_Y = np.sqrt(GS_diagonal_cython( Y_int, Y_length, psi_matrix, P, substring_length, delta, approximate))
        
        K = ((K/norm_Y).T/norm_X).T
    
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
        
        # Approximation of the kernel
        if argv[8].upper() == "TRUE":
            approximate = True
        elif argv[8].upper() == "FALSE":
            approximate = False
        else:
            raise ValueError
        
        output_file = argv[9] # Plain text output file name
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
    print "Approximate =", approximate
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
    K = gs_gram_matrix(X=X,
                            Y=Y,
                            amino_acid_property_file=amino_acid_property_file,
                            sigma_position=sigma_position,
                            sigma_amino_acid=sigma_amino_acid,
                            substring_length=substring_length,
                            approximate=approximate,
                            normalize_matrix=normalize_matrix)
    t_2 = time.time()
    
    print "Gram matrix computation took", t_2-t_1,"seconds."
    
    # Writing matrix in output file
    print "Writing matrix in output file ..."
    np.savetxt(output_file, K)
    
    print "\nSuccess!"
    
    
