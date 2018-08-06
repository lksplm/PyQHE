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

basis = BasisFermi(N=[5,5], m=[12,12])

diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j, s, p: i*(i==j)
H0 = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)

print("H0 hermitian: ",H0.is_hermitian())
int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m[0]), repeat=4) if j+k-l-m==0]

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

coeff = np.zeros((basis.m[0],basis.m[0],basis.m[0],basis.m[0]), dtype=np.float64)
for j, k, l, m in int_sites:
    coeff[j, k, l, m] = Vint(j, k, l, m)

Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
print("Hint hermitian: ",Hint.is_hermitian())

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
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