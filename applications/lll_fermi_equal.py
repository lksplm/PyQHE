import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.plotting import hinton_fast, energy_spin, spectrum_spin
from pyqhe.eigensystem import Eigensystem
import pickle


basis = BasisFermi(N=[2,2], m=[8,8])
#basis.print_states()

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

#Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
Hint= OperatorQuadCy(basis, site_indices=int_sites, spin_indices=[(0,1,1,0), (1,0,0,1)], \
                        op_func_site=coeff, op_func_spin=1.)
print("Hint hermitian: ",Hint.is_hermitian())

Hinteq = OperatorQuadCy(basis, site_indices=int_sites, spin_indices=[(0,0,0,0),], #, (1,1,1,1)
                        op_func_site=coeff, op_func_spin=1.)
print(Hinteq.matrix.nnz, Hinteq.matrix.max())
hinton_fast(Hinteq.dense)

print("Hinteq hermitian: ",Hinteq.is_hermitian())

p_sites = [(i,i+2) for i in range(basis.m[0]-2)]+[(i+2,i) for i in range(basis.m[0]-2)]
coeff_p = lambda i, j, s, p: np.sqrt((min(i,j)+1)*(min(i,j)+2))
Hpa = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(0,0)], op_func=coeff_p)
Hpb = OperatorLinCy(basis, site_indices=p_sites, spin_indices=[(1,1)], op_func=coeff_p)
Hp = Hpa + Hpb
print("Hp hermitian: ",Hp.is_hermitian())
print("Starting diagonalisation...", flush=True)

alpha = np.linspace(np.finfo(float).eps, 0.5, 100)
eps = np.linspace(np.finfo(float).eps, 0.1, 100)
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [0.25], eps], M=10)
eigsys = Eigensystem(ops_list=[H0, Hint, Hinteq], param_list=[[0.1], [0.25], [0.125]], M=30)

#energies, states = eigsys.energies, eigsys.states

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)

s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
print("S hermitian: ",S.is_hermitian())

print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint+Hinteq)
eigsys.add_observable(name="Einteq", op=Hinteq)
eigsys.add_observable(name="S", op=S)

L = eigsys.get_observable("L") #np.empty((self.M, *self.param_shape))
Einteq = eigsys.get_observable("Einteq")
Eint = eigsys.get_observable("Eint")
Sres = eigsys.get_observable("S")


spectrum_spin(L, Eint, Sres)
spectrum_spin(L, Einteq, Sres)

"""
_, ax = eigsys.plot_observable("E")
ax.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E$')

_, ax2 = eigsys.plot_observable("L", Mshow=3)
ax2.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$L$')

_, ax3 = eigsys.plot_observable("S", Mshow=3)
ax3.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel(r'$S$')
"""
plt.figure()

plt.plot(L[:,1:,:].flatten(), Eint[:,1:,:].flatten(), 'o')

# plt.plot(Ltots[idx[0],8], Energies[idx[0],8], 'ro')
#plt.xticks(np.arange(0, np.max(Ltots), 2))
plt.legend()
plt.xlabel('$L$')
plt.ylabel('$E_{int}/U$')
#plt.title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N_up, N_dwn, L_dwn))

plt.show()
"""
savedict = {'states': basis.states, 'Ltot': L, 'GndSts': eigsys.states, 'Esys': eigsys}
pickle.dump(savedict,open("results/result_pert_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), "wb" ))
"""