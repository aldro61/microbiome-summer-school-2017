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
'''
# **********************************************************
# To compile run:
# python build.py build_ext -b gs_kernel/
# **********************************************************

try:
    from numpy import get_include as get_numpy_include
except:
    print "Numpy is required to install this package. Please install Numpy and try again."
    exit()

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gs_kernel_cython", ["gs_kernel/gs_kernel_cython.pyx"], include_dirs=[get_numpy_include()])])
