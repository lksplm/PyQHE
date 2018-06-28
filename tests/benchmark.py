import numpy as np
from pyqhe.cython.hamiltonian_cy import quadratic_delta
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import Operator, OperatorQuad
from math import factorial, sqrt
from itertools import product
from time import time
N = 2
L = 6

def bench(N,L):
    L = np.int32(L)
    basis = BasisFermi([N,N],[L,L])
    #basis.print_states()

    def Vint(j, k, l, m):
        return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))


    vint = np.zeros([L]*4, dtype=np.float64)
    int_sites = [(j,k,l,m) for j,k,l,m in product(range(L), repeat=4) if j+k-l-m==0]

    for (j, k, l, m) in int_sites:
        vint[j, k, l, m] = Vint(j, k, l, m)

    #compare two methods
    t0=time()
    H_py = OperatorQuad(basis, site_indices=int_sites, spin_indices=[(0,1,1,0), (1,0,0,1)], op_func=Vint)
    tpy = time()-t0

    t0=time()
    H_cy = quadratic_delta(np.array(basis.basis, dtype=np.uint8), vint, L)
    tcy = time()-t0

    print("N={:d}, m={:d}, Py: {:.2f} s, Cy: {:.2f} s,  {:.1f}x".format(N, L, tpy, tcy, tpy/tcy))
    print((H_py.matrix!=H_cy.tocsc()).nnz)
    return tpy, tcy

Ns = [2, 3, 4]
Ls = [[4,6,8], [6, 8], [8]]

for a, N in enumerate(Ns):
    for b, L in enumerate(Ls[a]):
        bench(N,L)