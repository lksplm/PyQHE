import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast, spectrum_spin, energy_spin, spectrum_spin_mod, spectrum_spin_sz, energy_spin_sz
from pyqhe.eigensystem import Eigensystem, Observable
import pickle
from joblib import dump
from joblib import Memory

basis = BasisFermi(N=[3,3], m=[8,8], spin_conserved=False)

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

"""
p_sites = [(i,i+2) for i in range(basis.m[0]-2)]+[(i+2,i) for i in range(basis.m[0]-2)]
coeff_p = lambda i, j, s, p: np.sqrt((min(i,j)+1)*(min(i,j)+2))
Hpa = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(0,0)], op_func=coeff_p)
Hpb = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(1,1)], op_func=coeff_p)
Hp = Hpa + Hpb
del Hpa, Hpb
print("Hp hermitian: ",Hp.is_hermitian())
"""

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(1,1)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]
Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(0,1,0,1)], op_func_site=1., op_func_spin=1.)
Sc = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=0.25)
coeff_Sd = lambda s,u,v,w: (1. if s==0 else -1.)*(1. if u==0 else -1.)
Sd = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,0,1),(0,1,1,0),(0,0,0,0),(1,1,1,1)], op_func_site=0.25, op_func_spin=coeff_Sd)
coeff_Sz = lambda i, j, s, p: 0.5*(1. if s==0 else -1.)
Sz = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_Sz)
S = Sa + Sb + Sc + Sd + Sz
#S = Sb + Sd
Sop = Observable("S", S)
print("S hermitian: ",S.is_hermitian())
print("Sz hermitian: ",Sz.is_hermitian())
print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

print("Starting diagonalisation...", flush=True)

alpha = np.linspace(np.finfo(float).eps, 0.5, 20)
#eps = np.linspace(np.finfo(float).eps, 0.1, 100)
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [0.25], eps], M=10)
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], M=100)
#eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], M=100, simult_obs=Sop, simult_seed=[0j, 2j, 6j, 12j, 20j]) #[0.75j, 3.75j]
eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], simult_obs=Sop, full=True) #

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)
eigsys.add_observable(name="Sz", op=Sz)

#eigsys.add_observable(name="S", op=S)
#Modify spin
#Sz = 0.5*(basis.N[1]-basis.N[0])
eigsys.Observables["S"].eigenvalues = -0.5+np.sqrt(0.25+eigsys.Observables["S"].eigenvalues)

L = eigsys.get_observable("L")
Eall = eigsys.get_observable("E")
Eint = eigsys.get_observable("Eint")
Spin = eigsys.get_observable("S")
Spinz = eigsys.get_observable("Sz")

"""

_, ax4 = spectrum_spin_mod(L, Eint, Spin)
_, ax4 = spectrum_spin_mod(L, Eint, Spinz)
_, ax4 = spectrum_spin_sz(L, Eint, Spin, Spinz)
_, ax5 = energy_spin(alpha, Eall, Spin, Mshow=20, integer=True)
_, ax5 = energy_spin_sz(alpha, Eall, Spin, Spinz, Mshow=20, integer=True)

plt.show()
"""

savedict = {'states': basis.states, 'Esys': eigsys}
dump(savedict,"results/result_simult_ext_Sz_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), compress=3) #scaling
#pickle.dump(savedict,open("results/result_simult_imbal_{:d}__{:d}.p".format(*basis.m, *basis.N), "wb" ))
#_imbal_full