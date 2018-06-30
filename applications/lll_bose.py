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

N=6
basis = BasisBose(N=N, m=2*N)
#basis.print_states()

diag_sites = [(i,i) for i in range(basis.m)]
coeff_l = lambda i, j: i*(i==j)
H0 = OperatorLinCy(basis, site_indices=diag_sites, op_func=coeff_l)

print("H0 hermitian: ",H0.is_hermitian())
int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m), repeat=4) if j+k-l-m==0]

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))
coeff = np.zeros((basis.m,basis.m,basis.m,basis.m), dtype=np.float64)
for j, k, l, m in int_sites:
    coeff[j, k, l, m] = Vint(j, k, l, m)

Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
print("Hint hermitian: ",Hint.is_hermitian())

p_sites = [(i,i+2) for i in range(basis.m-2)]+[(i+2,i) for i in range(basis.m-2)]
coeff_p = lambda i, j: np.sqrt((min(i,j)+1)*(min(i,j)+2))
Hp = OperatorLinCy(basis, site_indices=p_sites,op_func=coeff_p)
Hp = Hp + H0
print("Hp hermitian: ",Hp.is_hermitian())
print("Starting diagonalisation...", flush=True)

alpha = np.linspace(np.finfo(float).eps, 0.5, 100)
eps = np.linspace(np.finfo(float).eps, 0.03, 20)
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [0.1], eps], M=10)
eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.62832]], M=10)
#eigsys = Eigensystem(ops_list=[H0, Hint, Hp], param_list=[alpha, [0.1], [0.01]], M=10)

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)

E = np.squeeze(eigsys.get_observable("E"))
L = eigsys.get_observable("L")
Eint = eigsys.get_observable("Eint")


_, ax = eigsys.plot_observable("E", Mshow=3)
ax.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E$')

_, ax2 = eigsys.plot_observable("L", Mshow=3)
ax2.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$L$')

plt.figure()
plt.plot(L[:,1:,:].flatten(), Eint[:,1:,:].flatten(), 'o')
plt.legend()
plt.xlabel('$L$')
plt.ylabel('$E_{int}/U$')
#plt.title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N_up, N_dwn, L_dwn))
"""
plt.figure()
#plt.pcolormesh(alpha, eps, -(E[1,:,:]-E[0,:,:]).T, cmap='jet')
plt.imshow( -(E[1,:,:]-E[0,:,:]).T, cmap='jet', interpolation='bicubic', origin='lower')
plt.colorbar()
plt.title(r'Gap $\Delta = E_1-E_0$')
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\epsilon$")
"""
plt.show()

savedict = {'states': basis.states, 'Ltot': L, 'Esys': eigsys}
pickle.dump(savedict,open("results/result_bose_{:d}_{:d}.p".format(basis.m, basis.N), "wb" ))