import numpy as np
import scipy.sparse as sp
from itertools import product, combinations, combinations_with_replacement
from scipy.special import binom
from pyqhe.hamiltonian import quadratic
from math import factorial, sqrt
#from util import hinton_fast
import time
N = [2,2]
m=[6,6]

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{}  {:.3f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed

def gen_basis(N, m):
    states_spin = [combinations(range(m), n) for n, m in zip(N, m)]
    Nstates_total = int(np.prod([int(binom(m, n)) for n, m in zip(N, m)]))
    m_total = int(np.sum(m))
    basis = np.zeros((Nstates_total, m_total), dtype=np.uint8)
    st_fermi = product(*states_spin)
    basis_l = np.zeros(Nstates_total, dtype=np.int)
    offs_arr  = np.insert(m[:-1], 0, 0)
    offs_arr_end = np.insert(m[1:], -1, -1)
    l_diag = np.concatenate([np.arange(n) for n in m])

    for i, idx in enumerate(st_fermi):
        s = np.zeros(m_total, dtype=np.uint8)
        for j, n in enumerate(m):
            np.add.at(s, np.array(idx[j]) + offs_arr[j], 1)

        basis[i, :] = s
        basis_l[i] = np.sum(s * l_diag)

    idx = np.argsort(basis_l)
    return basis[idx], basis_l[idx]

basis, _ = gen_basis(N,m)
print(basis.shape)

L_dwn=int(m[0])
vint = np.zeros([L_dwn]*4, dtype=np.float64)
idx = [(j,k,l,m) for j,k,l,m in product(range(L_dwn), repeat=4) if j+k-l-m==0]


def Vint_eq(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

for (j, k, l, m) in idx:
    vint[j, k, l, m] = Vint_eq(j, k, l, m)

@timeit
def compute_H(basis,vint):
    return quadratic(basis, vint, 6)

H = compute_H(basis, vint)

#hinton_fast(H.todense())