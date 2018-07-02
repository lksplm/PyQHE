import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisBose
from pyqhe.hamiltonian_bose import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast
from pyqhe.eigensystem import Eigensystem
import pickle

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

alpha = np.linspace(np.finfo(float).eps, 0.3, 100)#[np.finfo(float).eps] #np.linspace(np.finfo(float).eps, 1.0, 100)
Ns=[8]#3,4,5,6

for N in Ns:
    basis = BasisBose(N=N, m=2*N)

    diag_sites = [(i,i) for i in range(basis.m)]
    coeff_l = lambda i, j: i*(i==j)
    H0 = OperatorLinCy(basis, site_indices=diag_sites, op_func=coeff_l)

    print("H0 hermitian: ",H0.is_hermitian())
    int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m), repeat=4) if j+k-l-m==0]

    coeff = np.zeros((basis.m,basis.m,basis.m,basis.m), dtype=np.float64)
    for j, k, l, m in int_sites:
        coeff[j, k, l, m] = Vint(j, k, l, m)

    Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
    print("Hint hermitian: ",Hint.is_hermitian())

    print("Starting diagonalisation...", flush=True)


    eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.1]], M=10)


    eigsys.add_observable(name="L", op=H0)
    eigsys.add_observable(name="Eint", op=Hint)


    savedict = {'states': basis.states, 'Esys': eigsys}
    pickle.dump(savedict,open("results/result_bose_{:d}_{:d}.p".format(basis.m, basis.N), "wb" ))#Laughlin