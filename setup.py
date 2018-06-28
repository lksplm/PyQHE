from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('pyqhe.cython.hamiltonian_cy', ['pyqhe/cython/hamiltonian_cy.pyx'], include_dirs = [np.get_include()],language="c++"),
    ]


setup(
    name='PyQHE',
    version='0.1',
    packages=[''],
    url='',
    license='',
    author='Lukas Palm',
    author_email='',
    description='',
    ext_modules = cythonize(extensions,annotate=True)
)
