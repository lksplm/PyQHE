from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d, tensor_basis, spinful_fermion_basis_1d
from quspin.operators import quantum_operator, hamiltonian

import matplotlib.pyplot as plt
import numpy as np

from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLin, OperatorQuad
from pyqhe.util import hinton_fast

N=2
L=8
J=1.0
U=1.0
mu=0.1

basis = BasisFermi(N=[N, N], m=[L, L])
basis.print_states()


basis_qs = spinful_fermion_basis_1d(L, Nf=(N, N))


#basis matching
idx = []
for i in range(basis.Nbasis):
    v = basis.basis[i,:]
    s_up = "".join(map(str, v[0:L].tolist()))
    s_dwn = "".join(map(str, v[L:].tolist()))
    j = int(basis_qs.index(s_up,s_dwn))
    idx.append(j)

idx = np.argsort(idx)
basis.basis[:,:] = basis.basis[idx,:]
basis._build_lut()


# define site-coupling lists
hop_right = [[i, i + 1] for i in range(L - 1)]  # open boundarys
hop_left = [[i + 1, i] for i in range(L - 1)]  # open boundarys
hop_amp = lambda i, j: J

H_hop0 = OperatorLin(basis, site_indices=hop_right, spin_indices=[(0, 0), (1, 1)], op_func=hop_amp)
H_hop1 = OperatorLin(basis, site_indices=hop_left, spin_indices=[(0, 0), (1, 1)], op_func=hop_amp)  # h.c term
H_hop = H_hop0 + H_hop1

hop_right = [[-J, i, i + 1] for i in range(L - 1)]  # open boundarys
hop_left = [[+J, i, i + 1] for i in range(L - 1)]  # open boundarys

operator_list_0 = [
    ['+-|', hop_left],  # up hops left
    ['-+|', hop_right],  # up hops right
    ['|+-', hop_left],  # down hops left
    ['|-+', hop_right],  # down hops right
]

operator_dict = dict(H0=operator_list_0)

#H = quantum_operator(operator_dict, basis=basis_qs)
H_hop_qs = hamiltonian(static_list=operator_list_0, dynamic_list=[], basis=basis_qs)

f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
_, ax1 = hinton_fast(H_hop.matrix.todense(), ax=ax1)
_, ax2 =hinton_fast(np.real(H_hop_qs.todense()), ax=ax2)

assert (H_hop.matrix != H_hop_qs.tocsr()).nnz==0

pot = [[i, i] for i in range(L)]  # -\mu \sum_j n_{j \sigma}
pot_amp = lambda i, j: -mu
H_pot = OperatorLin(basis, site_indices=pot, spin_indices=[(0, 0), (1, 1)], op_func=pot_amp)

pot = [[-mu, i] for i in range(L)]  # -\mu \sum_j n_{j \sigma}
operator_list_1 = [
['n|', pot],  # up on-site potention
['|n', pot],  # down on-site potention
]
H_pot_qs = hamiltonian(static_list=operator_list_1, dynamic_list=[], basis=basis_qs)

f2, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
_, ax1 = hinton_fast(H_pot.matrix.todense(), ax=ax1)
_, ax2 =hinton_fast(np.real(H_pot_qs.todense()), ax=ax2)

assert (H_pot.matrix!=H_pot_qs.tocsr()).nnz == 0

interact = [[i, i, i, i] for i in range(L)]  # U/2 \sum_j n_{j,up} n_{j,down}
int_amp = lambda i, j, k, l: -U

H_int = OperatorQuad(basis, site_indices=interact, spin_indices=[(1, 0, 1, 0)], op_func=int_amp)
print("Hint hermitian: ", H_int.is_hermitian())

interact = [[U, i, i] for i in range(L)]  # U/2 \sum_j n_{j,up} n_{j,down}
operator_list_2 = [['n|n', interact]  # up-down interaction
]
H_int_qs = hamiltonian(static_list=operator_list_2, dynamic_list=[], basis=basis_qs)


f3, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
_, ax1 = hinton_fast(H_int.matrix.todense(), ax=ax1)
_, ax2 =hinton_fast(np.real(H_int_qs.todense()), ax=ax2)

assert (H_int.matrix!=H_int_qs.tocsr()).nnz == 0
plt.show()