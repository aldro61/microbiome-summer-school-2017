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
# To install:
# python setup.py install
# **********************************************************

try:
    from numpy import get_include as get_numpy_include
except:
    print "Numpy 1.6.2 or greater is required to install this package. Please install Numpy and try again."
    exit()

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup, find_packages, Extension

setup(
    name = "gs_kernel",
    version = "1.1",
    author="Sébastien Giguère",
    author_email="sebastien.giguere.8@ulaval.ca",
    description='A string kernel for small biomolecules.',
    license="GPL",
    keywords="string kernel",
    url="http://graal.ift.ulaval.ca/gs-kernel/",
    packages = find_packages(),

    # Dependencies
    install_requires = ['numpy>=1.6.2'],

    # Cython Extension
    ext_modules = [Extension("gs_kernel_cython", ["gs_kernel/gs_kernel_cython.c"], include_dirs=[get_numpy_include()])]
)
