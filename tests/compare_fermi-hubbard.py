from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d, tensor_basis, spinful_fermion_basis_1d
from quspin.operators import quantum_operator

import matplotlib.pyplot as plt
import numpy as np

from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLin, OperatorQuad
from pyqhe.util import hinton_fast
from pyqhe.eigensystem import Eigensystem

def solve_pyqhe(N=2, L=4, J=1.0, U=1.0, mu=0.01):
    basis = BasisFermi(N=[N,N], m=[L,L])

    # define site-coupling lists
    hop_right=[[i,i+1] for i in range(L-1)] #open boundarys
    hop_left=[[i+1,i] for i in range(L-1)] #open boundarys
    hop_amp = lambda i,j : -J

    H_hop0 = OperatorLin(basis, site_indices=hop_right, spin_indices=[(0,0), (1,1)], op_func=hop_amp)
    H_hop1 = OperatorLin(basis, site_indices=hop_left, spin_indices=[(0,0), (1,1)], op_func=hop_amp) #h.c term
    H_hop = H_hop0 + H_hop1


    pot=[[i,i] for i in range(L)] # -\mu \sum_j n_{j \sigma}
    pot_amp = lambda i,j : -mu
    H_pot = OperatorLin(basis, site_indices=pot, spin_indices=[(0,0), (1,1)], op_func=pot_amp)


    interact=[[i,i,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}
    int_amp = lambda i,j,k,l : -U

    Hint = OperatorQuad(basis, site_indices=interact, spin_indices=[(1,0,1,0)], op_func=int_amp)
    print("Hint hermitian: ",Hint.is_hermitian())


    alpha = np.linspace(-10, 10, 100)
    eigs = Eigensystem(ops_list=[H_hop, H_pot, Hint], param_list=[[1.0], [1.0], alpha], full=True)

    energies, states = eigs.energies, eigs.states

    print(energies.shape)
    plt.figure()
    for i in range(energies.shape[0]):
        plt.plot(alpha, energies[i,:].T)
    plt.title(r"Spectrum depending on $U$ for $J=1$")
    plt.xlabel(r'$U/J$')
    plt.ylabel(r'$E/J$')
    plt.show()

    return energies, states

def solve_quspin(N=2, L=4, J=1.0, U=1.0, mu=0.01):
    basis = spinful_fermion_basis_1d(L, Nf=(N, N))
    hop_right = [[-J, i, i + 1] for i in range(L - 1)]  # open boundarys
    hop_left = [[+J, i, i + 1] for i in range(L - 1)]  # open boundarys
    pot = [[-mu, i] for i in range(L)]  # -\mu \sum_j n_{j \sigma}
    interact = [[U, i, i] for i in range(L)]  # U/2 \sum_j n_{j,up} n_{j,down}

    operator_list_0=[
            ['+-|',hop_left],  # up hops left
            ['-+|',hop_right], # up hops right
            ['|+-',hop_left],  # down hops left
            ['|-+',hop_right], # down hops right
            ['n|',pot],        # up on-site potention
            ['|n',pot],        # down on-site potention
                                    ]

    operator_list_1 = [['n|n',interact]   # up-down interaction
                       ]

    operator_dict = dict(H0=operator_list_0, H1=operator_list_1)

    H = quantum_operator(operator_dict, basis=basis)

    H_lmbda1 = H.tohamiltonian(pars=dict(H0=0.5, H1=1.0))
    ev, es = H_lmbda1.eigh()

    M = ev.shape[0]
    alpha = np.linspace(-10, 10, 100)
    energies = np.empty((M, alpha.shape[0]))  # allocate array for results
    eigenstate = np.empty((M, M, alpha.shape[0]), dtype=np.complex64)
    for i, u in enumerate(alpha):  # scan different u values
        H_lmbda1 = H.tohamiltonian(pars=dict(H0=1.0, H1=u))
        ev, es = H_lmbda1.eigh()
        #ev, es = H_lmbda1.eigsh(k=M, which='LR', sigma=0.0)
        energies[:, i] = ev
        eigenstate[:,:,i] = es

    plt.figure()
    for i in range(energies.shape[0]):
        plt.plot(alpha, energies[i,:])
    plt.title(r"Spectrum depending on $U$ for $J=1$")
    plt.xlabel(r'$U/J$')
    plt.ylabel(r'$E/J$')
    plt.show()

    return energies, np.real(eigenstate)

e1, s1 = solve_pyqhe()
e2, s2 = solve_quspin()


alpha = np.linspace(-10, 10, 100)
for i in range(e1.shape[0]):
    plt.plot(alpha, e1[i, :]-e2[i, :])
plt.title(r"Spectrum depending on $U$ for $J=1$")
plt.xlabel(r'$U/J$')
plt.ylabel(r'$E/J$')
plt.show()

print(np.allclose(e1,e2))
print(np.allclose(s1,np.swapaxes(s2,axis1=0, axis2=1)))

print(s1.shape)
print(s2.shape)

plt.figure()
for i in range(s1.shape[0]):
    print(np.sort(s1[:,i,50]))
    print(np.sort(s2[:,i,50]))


    plt.plot(np.zeros(s1.shape[0])+2*i,np.sort(s1[:,i,50]), 'r*')
    plt.plot(np.zeros(s1.shape[0])+1+2*i,np.sort(s2[:,i,50]), 'b*')
plt.show()