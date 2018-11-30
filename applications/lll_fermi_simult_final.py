import sys, os
#sys.path.append(os.path.abspath('../../PyQHE/'))
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast
from pyqhe.eigensystem import Eigensystem, Observable
#import pickle
from joblib import dump
from joblib import Memory

cachedir = 'cache'
if not os.path.isdir(cachedir): os.mkdir(cachedir)
memory = Memory(cachedir=cachedir, verbose=True)

basis = BasisFermi(N=[3,3], m=[8,8])

@memory.cache
def comp_H0(basis):
    diag_sites = [(i,i) for i in range(basis.m[0])]
    coeff_l = lambda i, j, s, p: i*(i==j)
    return OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)

@memory.cache
def comp_H1(basis):
    int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m[0]), repeat=4) if j+k-l-m==0]
    def Vint(j, k, l, m):
        return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))
    coeff = np.zeros((basis.m[0],basis.m[0],basis.m[0],basis.m[0]), dtype=np.float64)
    for j, k, l, m in int_sites:
        coeff[j, k, l, m] = Vint(j, k, l, m)

    return OperatorQuadDeltaCy(basis, coeff=coeff)

@memory.cache
def comp_H2(basis):
    diag_sites = [(i, i) for i in range(basis.m[0])]
    Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
    s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

    Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
    return Sa + Sb

H0 = comp_H0(basis)
print("H0 hermitian: ",H0.is_hermitian())

Hint = comp_H1(basis)
print("Hint hermitian: ",Hint.is_hermitian())

S = comp_H2(basis)
print("S hermitian: ",S.is_hermitian())
Spin = Observable("S", S)

print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

print("Starting diagonalisation...", flush=True)

alpha = np.linspace(np.finfo(float).eps, 0.4, 4)

seeds = [1j*a*(a+1) for a in range(10)]
seed = seeds[0:(basis.N[0]+1)]
print(seed)

eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [.25]], M=100, simult_obs=Spin, simult_seed=seed)

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)

savedict = {'states': basis.states, 'Esys': eigsys}
#pickle.dump(savedict,open("results/result_simult_full_gap_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), "wb" ))
dump(savedict,"results/result_simult_final_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), compress=3) #scaling