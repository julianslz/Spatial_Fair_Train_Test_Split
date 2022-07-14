from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext = Extension("cython_kriging",
            sources=["cython_kriging.pyx"],
            language="c++",
            include_dirs=[],
            libraries=[],
            extra_link_args=[])

setup(ext_modules = cythonize([ext]), include_dirs=[numpy.get_include()])
