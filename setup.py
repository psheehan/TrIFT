import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

import numpy
import pybind11

trift = Extension("trift.trift",sources=["src/trift.cc"],\
        include_dirs=["./include","./delaunator-cpp/include",\
        pybind11.get_include(False), pybind11.get_include(True)], \
        language="c++", \
        extra_compile_args=['-std=c++11','-stdlib=libc++','-Ofast',\
        '-march=native','-mmacosx-version-min=10.7'])

setup(name="trift", version="0.9.0", packages=["trift"], ext_modules=[trift])
