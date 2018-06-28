import matplotlib.pyplot as plt
import numpy as np

from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLin, OperatorQuad, Solve
from pyqhe.util import hinton_fast

L=4
J=1.0
U=1.0
mu=0.01

basis = BasisFermi(N=[2,2], m=[L,L])
basis.print_states()

# define site-coupling lists
hop_right=[[i,i+1] for i in range(L-1)] #open boundarys
hop_left=[[i+1,i] for i in range(L-1)] #open boundarys
hop_amp = lambda i,j : -J


H_hop0 = OperatorLin(basis, site_indices=hop_right, spin_indices=[(0,0), (1,1)], op_func=hop_amp)
H_hop1 = OperatorLin(basis, site_indices=hop_left, spin_indices=[(0,0), (1,1)], op_func=hop_amp) #h.c term
H_hop = H_hop0 + H_hop1

print(H_hop.matrix)
hinton_fast(H_hop.dense)
print("H0 hermitian: ",H_hop.is_hermitian())

pot=[[i,i] for i in range(L)] # -\mu \sum_j n_{j \sigma}
pot_amp = lambda i,j : -mu
H_pot = OperatorLin(basis, site_indices=pot, spin_indices=[(0,0), (1,1)], op_func=pot_amp)


interact=[[i,i,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}
int_amp = lambda i,j,k,l : -U

Hint = OperatorQuad(basis, site_indices=interact, spin_indices=[(1,0,1,0)], op_func=int_amp)
print("Hint hermitian: ",Hint.is_hermitian())

hinton_fast(Hint.dense)

alpha = np.linspace(-10, 10, 100)
energies, states = Solve(ops_list=[H_hop, H_pot, Hint], param_list=[[1.0], [1.0], alpha], full=True)

print(energies.shape)
plt.figure()
for i in range(energies.shape[0]):
    plt.plot(alpha, energies[i,0,0,:].T)
plt.title(r"Spectrum depending on $U$ for $J=1$")
plt.xlabel(r'$U/J$')
plt.ylabel(r'$E/J$')
plt.show()

