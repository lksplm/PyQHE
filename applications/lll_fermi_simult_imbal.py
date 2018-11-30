import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast, spectrum_spin, energy_spin
from pyqhe.eigensystem import Eigensystem, Observable
import pickle

basis = BasisFermi(N=[3,4], m=[10,10])
Sz = 0.5*(basis.N[1]-basis.N[0])

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
Sop = Observable("S", S)
print("S hermitian: ",S.is_hermitian())
print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

print("Starting diagonalisation...", flush=True)

alpha = np.linspace(np.finfo(float).eps, 0.4, 10)
seeds = [1j*((a/2)*(a/2+1)-(Sz*(Sz+1))) for a in range(1,10,2)]
seed = seeds[0:(basis.N[1])]
print(seed)

eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], M=100, simult_obs=Sop, simult_seed=seed) #[0.75j, 3.75j]
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], simult_obs=Sop, full=True) #

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)

#Modify spin
print(np.unique(np.round(eigsys.Observables["S"].eigenvalues,3)))
eigsys.Observables["S"].eigenvalues = -0.5+np.sqrt(0.25+(eigsys.Observables["S"].eigenvalues+Sz*(Sz+1)))
print(np.unique(np.round(eigsys.Observables["S"].eigenvalues,3)))

savedict = {'states': basis.states, 'Esys': eigsys}
pickle.dump(savedict,open("results/result_simult_imbal_{:d}_{:d}_{:d}_{:d}.p".format(*basis.N, *basis.m), "wb" ))
