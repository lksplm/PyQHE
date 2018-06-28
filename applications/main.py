import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt, factorial
from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLin, OperatorQuad, Solve
#from util import hinton_nolabel, hinton_fast

basis = BasisFermi(N=[2,2], m=[8,8])
#basis.print_states()

diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j: i*(i==j)
H0 = OperatorLin(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)

#hinton_fast(H0.dense)

print("H0 hermitian: ",H0.is_hermitian())
int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m[0]), repeat=4) if j+k-l-m==0]


def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

Hint = OperatorQuad(basis, site_indices=int_sites, spin_indices=[(0,1,1,0), (1,0,0,1)], op_func=Vint)
print("Hint hermitian: ",Hint.is_hermitian())

#hinton_nolabel(Hint.dense)

alpha = np.linspace(0.01, 0.5, 30)
energies, states = Solve(ops_list=[H0, Hint], param_list=[[0.25], alpha], M=5, full=True)

print(energies.shape)
plt.figure()
for i in range(4):
    plt.plot(alpha, energies[i,:].T)
plt.show()