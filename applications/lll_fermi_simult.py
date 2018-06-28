import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadCy, OperatorQuadDeltaCy
from pyqhe.util import hinton_fast
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
print("Starting diagonalisation...", flush=True)

Sa = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0)], op_func=1.)
s_sites = [(p,k,k,p) for k,p in product(range(basis.m[0]), repeat=2)]

Sb = OperatorQuadCy(basis, site_indices=s_sites, spin_indices=[(1,0,1,0)], op_func_site=1., op_func_spin=1.)
S = Sa + Sb
print("S hermitian: ",S.is_hermitian())
Spin = Observable("S", S)

print("[L, S^2] = 0: ",S.commutes(H0))
print("[Hint, S^2] = 0: ",S.commutes(Hint))

#alpha = np.linspace(np.finfo(float).eps, 0.5, 30)
alpha = np.linspace(-1., 1., 100)

eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [-0.25]], M=300, simult_obs=Spin, simult_seed=[2j, 6j])

eigsys.add_observable(name="L", op=H0)
eigsys.add_observable(name="Eint", op=Hint)
#eigsys.add_observable(name="S", op=S)

L = eigsys.get_observable("L") #np.empty((self.M, *self.param_shape))
Eint = eigsys.get_observable("Eint")


_, ax = eigsys.plot_observable("E")
ax.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E$')

_, ax2 = eigsys.plot_observable("L", Mshow=3)
ax2.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$L$')

_, ax3 = eigsys.plot_observable("S")
ax3.set_title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel(r'$S$')

plt.figure()

plt.plot(L[:,1:,:].flatten(), Eint[:,1:,:].flatten(), 'o')

# plt.plot(Ltots[idx[0],8], Energies[idx[0],8], 'ro')
#plt.xticks(np.arange(0, np.max(Ltots), 2))
plt.legend()
plt.xlabel('$L$')
plt.ylabel('$E_{int}/U$')
#plt.title('$N_{{\\uparrow={:d} }}, N_{{\\downarrow={:d} }}, m={:d}$'.format(N_up, N_dwn, L_dwn))

S_res = eigsys.get_observable("S")

Splt = S_res.copy()
Splt[Splt < 1.e-5] = 0
Splt[np.isnan(Splt)] = -1
Splt = np.array(np.round(Splt), dtype=np.int)
Slbl = np.unique(Splt)
print(Slbl)
Splt2 = Splt.copy()
for i, spi in enumerate(Slbl):
    Splt[Splt2 == spi] = i

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker = ['^', 'v', '*']
cmap = ListedColormap(cycle)
norm = BoundaryNorm(np.arange(len(Slbl) + 1) - 0.5, ncolors=len(Slbl))

legend_elements = [plt.Line2D([0], [0], color=cycle[i], marker=marker[i], label='S={:d}'.format(s)) for i, s in
                   enumerate(Slbl)]
plt.figure()
for i, s in enumerate(Slbl):
    idx = (Splt == i)
    plt.scatter(L[idx], Eint[idx], c=Splt[idx], marker=marker[i], facecolor='none', alpha=0.5, cmap=cmap,
                norm=norm)
plt.legend(handles=legend_elements)
plt.show()


#savedict = {'states': basis.states, 'Ltot': L, 'GndSts': eigsys.states, 'Esys': eigsys}
#pickle.dump(savedict,open("results/result_pert_{:d}_{:d}.p".format(basis.m[0], basis.N[0]), "wb" ))