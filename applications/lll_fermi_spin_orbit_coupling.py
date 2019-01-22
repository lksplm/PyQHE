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

basis = BasisFermi(N=[2,2], m=[8,8])

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
Hp = Hpa + Hpb

print("Hp hermitian: ",Hp.is_hermitian())

coeff_soa = lambda i, j, s, p: 0.5*i*(1. if s==0 else -1.)
Hsoa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0),(1,1)], op_func=coeff_soa)
so_sites = [(p,k,p,k) for p,k in product(range(basis.m[0]), repeat=2)]
#so_spins = [(0,0,0,0),(0,1,0,1),(1,0,1,0),(1,1,1,1)]
so_spins = [(0,1,1,0),(1,0,0,1)]
coeff_sob_site = lambda i,j,k,l: -0.5*i
coeff_sob_spin = lambda s,u,v,w: (1. if u==0 else -1.)
Hsob = OperatorQuadCy(basis, site_indices=so_sites, spin_indices=so_spins, op_func_site=coeff_sob_site, op_func_spin=coeff_sob_spin)
Hso = Hsoa + Hsob
print(Hso.matrix.nnz)
hinton_fast(Hsoa.dense)
hinton_fast(Hsob.dense)

plt.show()
print("Hso hermitian: ",Hso.is_hermitian())

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]
Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
Sop = Observable("S", S)
print("S hermitian: ",S.is_hermitian())
print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

print("Starting diagonalisation...", flush=True)

seeds = [1j*a*(a+1) for a in range(10)]
seed = seeds[0:(basis.N[0]+1)]
print(seed)

alpha = np.linspace(np.finfo(float).eps, 0.5, 50)
#eps = np.linspace(np.finfo(float).eps, 0.1, 100)

#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [0.25], [0.05]], M=50, simult_obs=Sop, simult_seed=seeds)
eigsys = Eigensystem(ops_list=[H0, Hint, Hso, Hp], param_list=[alpha, [0.25], [0.05], [0.0]], M=50)


eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)
eigsys.add_observable(name="S", op=S)

#Modify spin
Sz = 0.5*(basis.N[1]-basis.N[0])
eigsys.Observables["S"].eigenvalues = -0.5+np.sqrt(0.25+(eigsys.Observables["S"].eigenvalues+Sz*(Sz+1)))

L = eigsys.get_observable("L")
Eall = eigsys.get_observable("E")
Eint = eigsys.get_observable("Eint")
Spin = eigsys.get_observable("S")


_, ax = eigsys.plot_observable("E")
ax.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E$')
"""
_, ax2 = eigsys.plot_observable("L")
ax2.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$L$')

_, ax3 = eigsys.plot_observable("S", Mshow=10)
ax3.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel(r'$S$')

#print(Spin)

_, ax4 = spectrum_spin(L, Eint, Spin)
"""
_, ax5 = energy_spin(alpha, Eall, Spin, Mshow=10)

plt.show()
"""
savedict = {'states': basis.states, 'Esys': eigsys}
pickle.dump(savedict,open("results/result_simult_imbal_{:d}_{:d}_{:d}_{:d}.p".format(*basis.N, *basis.m), "wb" ))
#_imbal_full
"""

