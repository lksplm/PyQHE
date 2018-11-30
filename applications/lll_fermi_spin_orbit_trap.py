import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast, energy_spin, spectrum_spin
from pyqhe.eigensystem import Eigensystem, Observable
import pickle

basis = BasisFermi(N=[2,2], m=[8,8])

diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j, s, p: i*(i==j)

#H0 = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)
H0a = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0),], op_func=coeff_l)
H0b = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(1,1),], op_func=coeff_l)

H0 = H0a + H0b
print("H0 hermitian: ",H0a.is_hermitian())
print("H0 hermitian: ",H0b.is_hermitian())
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
Hp = Hpa + Hpb
print("Hp hermitian: ",Hp.is_hermitian())
print("Starting diagonalisation...", flush=True)

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
print("S hermitian: ",S.is_hermitian())
Spin = Observable("S", S)

#print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

omega = np.linspace(0.5, 1.0-np.finfo(float).eps, 100)
xsi = 0.1
alpha = 1-omega
alphab = 1+xsi-omega#/(xsi+1)

seeds = [1j*a*(a+1) for a in range(10)]
seed = seeds[0:(basis.N[0]+1)]
print(seed)

param_array = np.empty((alpha.shape[0]),dtype=object)
for i, a in enumerate(alpha):
    param_array[i] = [alpha[i],alphab[i],0.25, 0.08]

#eigsys = Eigensystem(ops_list=[H0a, H0b, Hint], param_list=param_array, M=30, simult_obs=Spin, simult_seed=seed)
eigsys = Eigensystem(ops_list=[H0a, H0b, Hint, Hp], param_list=param_array, full=True, simult_obs=Spin)
#eigsys = Eigensystem(ops_list=[H0a, H0b, Hint, Hp], param_list=param_array, M=30, simult_obs=Spin, simult_seed=seed)
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [.25]], M=100, simult_obs=Spin, simult_seed=seed)
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [.25]], M=50, simult_obs=Spin, simult_seed=[0j, 2j, 6j, 12j, 20j])
#eigsys = Eigensystem(ops_list=[H0a, H0b, Hint], param_list=[alpha, alphab, [.25]], full=True, simult_obs=Spin)

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)


L = eigsys.get_observable("L") #np.empty((self.M, *self.param_shape))
Eint = eigsys.get_observable("Eint")
Sres = eigsys.get_observable("S")
E = eigsys.get_observable("E")

Esort = E.copy()
Ssort = Sres.copy()
for i in range(E.shape[1]):
    idx = np.argsort(Esort[:,i])
    Esort[:,i] = Esort[idx,i]
    Ssort[:,i] = Ssort[idx,i]

energy_spin(alpha, Esort, Ssort)
spectrum_spin(L, Eint, Ssort)
#spectrum_spin(L, Esort, Ssort)

plt.show()

"""

savedict = {'states': basis.states, 'Esys': eigsys}
pickle.dump(savedict,open("results/result_simult_full_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), "wb" ))

"""