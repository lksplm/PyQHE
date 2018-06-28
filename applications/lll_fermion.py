import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadDeltaCy, Solve
from pyqhe.util import hinton_fast

basis = BasisFermi(N=[2,2], m=[8,8])
#basis.print_states()

diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j, s, p: i*(i==j)
H0 = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)

#hinton_fast(H0.dense)

print("H0 hermitian: ",H0.is_hermitian())
int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m[0]), repeat=4) if j+k-l-m==0]

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

coeff = np.zeros((basis.m[0],basis.m[0],basis.m[0],basis.m[0]), dtype=np.float64)
for j, k, l, m in int_sites:
    coeff[j, k, l, m] = Vint(j, k, l, m)

Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
print("Hint hermitian: ",Hint.is_hermitian())

#hinton_fast(Hint.dense)

alpha = np.linspace(0.01, 0.5, 30)
energies, states = Solve(ops_list=[H0, Hint], param_list=[alpha, [0.25]], M=5)

print(energies.shape)
plt.figure()
for i in range(5):
    plt.plot(alpha, energies[i,:,0])

plt.title(r"Spectrum depending on $\alpha$ for $\eta=0.25$")
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$E$')
plt.show()