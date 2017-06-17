#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
---------------------------------------------------------------------
Copyright 2011, 2012, 2013 Sébastien Giguère

This example is provided with the GSkernel

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

Author: Alexandre Drouin
Email: alexandre.drouin.8@ulaval.ca

This example shows how we can attempt to predict peptide binding
affinity to proteins using an SVM classifier with the GS Kernel.

Data:
The data was provided by the Dana-Farber Repository for Machine Learning in Immunology
'''

from os.path import *
from GSkernel_fast import GS_gram_matrix_fast

try:
    from sklearn.svm import SVC
except:
    print 'This example requires Scikit-learn. You can download it from: http://scikit-learn.org/stable/install.html'

try:
    import numpy as np
except:
    print 'This example requires Numpy. You can download it from: http://scipy.org/Download'


def load_dataset(dsDirectory, allele):
    '''
    Loads the dataset containing binary information on
    peptide binding to MHC alleles
    
    dsDirectory -- The directory containing the datasets
    allele -- The allele for which to load the data
    '''
    datasets = [join(dsDirectory, allele + "_binding.dat"),
                join(dsDirectory, allele + "_nonbinding.dat")]
    binding = []
    peptides = []
    for dsPath in datasets:
        f = open(expandvars(dsPath))
        lines = f.readlines()
        f.close()
        for line in lines:
            spt = line.split()
            peptides.append(spt[2].upper())
            if "_binding" in dsPath:
                binding.append(1)
            else:
                binding.append(-1)
    return {'sequences':np.array(peptides), 
            'classes':np.array(binding)}

if __name__=="__main__":
    print 'Loading dataset...'
    dataset = load_dataset('data', 'HLA-B5701')
    print 'Loaded ', len(dataset['classes']), ' examples!'
    
    print
    print 'Computing gram matrix...'
    train_matrix = GS_gram_matrix_fast(X=dataset['sequences'],
                                  Y=dataset['sequences'],
                                  amino_acid_property_file='../amino_acids_matrix/AA.blosum50.dat',
                                  sigma_position=1.0,
                                  sigma_amino_acid=1.0,
                                  substring_length=2,
                                  normalize_matrix=True)
    
    print
    print 'Fitting an SVM classifier...'
    print len(dataset['sequences'])
    estimator = SVC(kernel='precomputed')
    estimator.fit(X=train_matrix, y=dataset['classes'])
    
    print 'Done!'
    
    
