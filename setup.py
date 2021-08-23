from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize('cython_kriging.pyx'),
    include_dirs=[np.get_include()],
    requires=['Cython', 'numpy']
)
# python setup.py build_ext --inplace
# Compiling the C code
# 1. Open the terminal from Anaconda
# 2. Type: cd (directory where your pyx file is
# 3. Run:
# python setup.py build_ext --inplace
