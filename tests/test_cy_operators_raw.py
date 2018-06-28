import numpy as np
from pyqhe.cython.hamiltonian_cy import quadratic_delta, quadratic, linear
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import Operator, OperatorQuad, OperatorLin
from math import factorial, sqrt
from itertools import product
from time import time
N = 2
L = 6

L = np.uint32(L)
basis = BasisFermi([N,N],[L,L])
#basis.print_states()
"""
Linear Hamiltonian 
"""
diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j: i*(i==j)
coeff = np.empty((L,L,2,2), dtype=np.float64)
coeff[...] = np.arange(L)[:,None,None,None]

#Python
H0_py = OperatorLin(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)
#Cython
H0_cy = linear(np.array(basis.basis, dtype=np.uint8), np.array(diag_sites, dtype=np.uint32),\
               np.array([(0,0), (1,1)], dtype=np.uint32), coeff, np.uint32(L))

assert (H0_py.matrix!=H0_cy.tocsc()).nnz == 0


"""
Interaction Hamiltonian 
"""
def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

vint = np.zeros([L]*4, dtype=np.float64)
int_sites = [(j,k,l,m) for j,k,l,m in product(range(L), repeat=4) if j+k-l-m==0]

for (j, k, l, m) in int_sites:
    vint[j, k, l, m] = Vint(j, k, l, m)

#Python
H1_py = OperatorQuad(basis, site_indices=int_sites, spin_indices=[(0,1,1,0), (1,0,0,1)], op_func=Vint)
#Cython 1
H1_cy = quadratic_delta(np.array(basis.basis, dtype=np.uint8), vint, L)
#Cython 2
#int_spins = np.empty((2,2,2,2), dtype=np.float64)

H1_cy2 = quadratic(np.array(basis.basis, dtype=np.uint8), np.array(int_sites, dtype=np.uint32), \
                   np.array([(0, 1, 1, 0), (1, 0, 0, 1)], dtype=np.uint32), vint, np.ones((2,2,2,2), dtype=np.float64) ,L)


assert (H1_py.matrix!=H1_cy.tocsc()).nnz == 0
assert (H1_py.matrix!=H1_cy2.tocsc()).nnz == 0
assert (H1_cy.tocsc()!=H1_cy2.tocsc()).nnz == 0

print("All Operators work the same!")
