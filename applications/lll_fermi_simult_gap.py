import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast
from pyqhe.eigensystem import Eigensystem, Observable
#import pickle
from joblib import dump

basis = BasisFermi(N=[3,3], m=[8,8])

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

p_sites = [(i,i+2) for i in range(basis.m[0]-2)]+[(i+2,i) for i in range(basis.m[0]-2)]
coeff_p = lambda i, j, s, p: np.sqrt((min(i,j)+1)*(min(i,j)+2))
Hpa = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(0,0)], op_func=coeff_p)
Hpb = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(1,1)], op_func=coeff_p)

coeff_pc = lambda i, j, s, p:  i+1
Hpc = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_pc)

Hp = Hpa + Hpb + Hpc
print("Hp hermitian: ",Hp.is_hermitian())
print("Starting diagonalisation...", flush=True)

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
print("S hermitian: ",S.is_hermitian())
Spin = Observable("S", S)

print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

alpha = np.linspace(np.finfo(float).eps, 0.5, 100)#0.5
eps = np.linspace(np.finfo(float).eps, 0.1, 50)

#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [.25], eps], full=True, simult_obs=Spin)
eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [.125, .25, 0.5, 1.0], [0.02]], M=50, simult_obs=Spin, simult_seed=[0j, 2j, 6j, 12j])
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [.25], eps], M=50, simult_obs=Spin, simult_seed=[0j, 2j, 6j])
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [.25], eps], M=10, simult_obs=Spin, simult_seed=[2j])
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [.25]], M=50, simult_obs=Spin, simult_seed=[0j, 2j, 6j, 12j, 20j])
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [.25]], full=True, simult_obs=Spin)
"""
E = np.squeeze(eigsys.get_observable("E"))

plt.figure()
for i in range(20):
    plt.plot(alpha, np.sort(E[i,:,-1], axis=0))

plt.figure()
#plt.pcolormesh(alpha, eps, -(E[1,:,:]-E[0,:,:]).T, cmap='jet')
plt.imshow( -(E[1,:,:]-E[0,:,:]).T, cmap='viridis_r', interpolation='bicubic', origin='lower')
plt.colorbar()
plt.title(r'Gap $\Delta = E_1-E_0$')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\epsilon$")
plt.show()
"""

savedict = {'states': basis.states, 'Esys': eigsys}
#pickle.dump(savedict,open("results/result_simult_full_gap_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), "wb" ))
dump(savedict,"results/result_simult_gap_sclaing_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), compress=3) #scaling