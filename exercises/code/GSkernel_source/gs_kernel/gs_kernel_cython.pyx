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
'''

import cython
import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float64_t FLOAT64_t
ctypedef np.int8_t INT8_t
ctypedef np.int64_t INT64_t


cdef inline INT64_t int_max(INT64_t a, INT64_t b): return a if a >= b else b
cdef inline INT64_t int_min(INT64_t a, INT64_t b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t GSKernelCython(INT8_t[::1] x1,
                                     INT64_t l1,
                                     INT8_t[::1] x2,
                                     INT64_t l2, 
                                     FLOAT64_t[:,::1] psiMat, 
                                     FLOAT64_t[:,::1] P, 
                                     INT64_t L):

    cdef INT64_t i,j,l, maxL
    cdef FLOAT64_t tmp, b, cumB
    
    cumB = 0.0
    for i in range(l1):
        maxL = int_min(L, l1-i)
        
        for j in range(l2):
            if l2-j < maxL:
                maxL = (l2-j)
            
            tmp = 1.0
            b = 0.0
            
            for l in range(maxL):
                tmp *= psiMat[ x1[i+l], x2[j+l] ]
                b += tmp
                
            cumB += P[i,j] * b
            
    return cumB


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t GSKernelCythonApprox(INT8_t[::1] x1,
                                           INT64_t l1, 
                                           INT8_t[::1] x2,
                                           INT64_t l2, 
                                           FLOAT64_t[:,::1] psiMat, 
                                           FLOAT64_t[:,::1] P, 
                                           INT64_t L,
                                           INT64_t delta):

    cdef INT64_t i,j,l, maxL
    cdef FLOAT64_t tmp, b, cumB
    
    cumB = 0.0
    for i in range(l1):
        maxL = int_min(L, l1-i)
        
        for j in range(int_max(0, i-delta+1), int_min(l2, i+delta)):
            if l2-j < maxL:
                maxL = (l2-j)
            
            tmp = 1.0
            b = 0.0
            
            for l in range(maxL):
                tmp *= psiMat[ x1[i+l], x2[j+l] ]
                b += tmp
                
            cumB += P[i,j] * b
            
    return cumB

@cython.boundscheck(False)
@cython.wraparound(False)
def GS_gram_matrix_cython(INT8_t[:,::1] X,
                          INT64_t[::1] X_length,
                          INT8_t[:,::1] Y,
                          INT64_t[::1] Y_length,
                          FLOAT64_t[:,::1] psiMat, 
                          FLOAT64_t[:,::1] P,
                          INT64_t L,
                          INT64_t delta,
                          symetric,
                          approximate):
    
    cdef int i,j
    cdef FLOAT64_t[:,::1] K = np.zeros( (X.shape[0], Y.shape[0]), dtype=np.float64 )
    
    if symetric:
        if approximate:
            for i in range(X.shape[0]):
                K[i,i] = GSKernelCythonApprox( X[i], X_length[i], X[i], X_length[i], psiMat, P, L, delta)
                for j in range(i):
                    K[i,j] = GSKernelCythonApprox( X[i], X_length[i], X[j], X_length[j], psiMat, P, L, delta)
                    K[j,i] = K[i,j]
        else:
            for i in range(X.shape[0]):
                K[i,i] = GSKernelCython( X[i], X_length[i], X[i], X_length[i], psiMat, P, L )
                for j in range(i):
                    K[i,j] = GSKernelCython( X[i], X_length[i], X[j], X_length[j], psiMat, P, L )
                    K[j,i] = K[i,j]
            
    else:
        if approximate:
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    K[i,j] = GSKernelCythonApprox( X[i], X_length[i], Y[j], Y_length[j], psiMat, P, L, delta)
        else:
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    K[i,j] = GSKernelCython( X[i], X_length[i], Y[j], Y_length[j], psiMat, P, L )

    return np.asarray(K)

@cython.boundscheck(False)
@cython.wraparound(False)
def GS_diagonal_cython( INT8_t[:,::1] X,
                    INT64_t[::1] X_length,
                    FLOAT64_t[:,::1] psiMat, 
                    FLOAT64_t[:,::1] P,
                    INT64_t L,
                    INT64_t delta,
                    approximate):
    
    cdef int i
    cdef FLOAT64_t[::1] K = np.zeros( X.shape[0], dtype=np.float64 )
    
    if approximate:
        for i in range(X.shape[0]):
            K[i] = GSKernelCythonApprox( X[i], X_length[i], X[i], X_length[i], psiMat, P, L, delta)
    else:
        for i in range(X.shape[0]):
            K[i] = GSKernelCython( X[i], X_length[i], X[i], X_length[i], psiMat, P, L )
    
    return np.asarray(K)
