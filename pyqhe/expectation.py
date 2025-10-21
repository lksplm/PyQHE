from .basis import *
from .cython.hamiltonian_cy import *
from .cython import hamiltonian_bose_cy
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fermionic expectation values
def expect_lin(state, basis, a, b):
    return expectation_lin(state.astype(np.complex64), basis.states, np.int32(a), np.int32(b), np.int32(basis.Lbasis//2))

def expect_quad(state, basis, a, b, c, d):
    return expectation_quad_test(state.astype(np.complex64), basis.states, basis.lookup_map, np.int32(a), np.int32(b),\
                                 np.int32(c), np.int32(d), np.int32(basis.Lbasis//2))

def expect_six(state, basis, a, b, c, d, e, f):
    return expectation_six_test(state.astype(np.complex64), basis.states, basis.lookup_map, np.int32(a), np.int32(b),\
                                 np.int32(c), np.int32(d), np.int32(e), np.int32(f), np.int32(basis.Lbasis//2))

# Bosonic expectation values
def expect_lin_bose(state, basis):
    """
    Compute 1-particle bosonic density matrix ρ_ij = <ψ|b†_i b_j|ψ>

    :param state: quantum state vector (will be converted to complex64)
    :param basis: BasisBose object containing basis states and lookup map
    :return: complex matrix of shape [m, m] where m is the number of modes
    """
    return hamiltonian_bose_cy.expect_lin(
        state.astype(np.complex64),
        np.array(basis.basis, dtype=np.uint8),
        basis.lookup_map,
        basis.N
    )

def expect_quad_bose(state, basis):
    """
    Compute 2-particle bosonic density matrix ρ_ijkl = <ψ|b†_i b†_j b_k b_l|ψ>

    :param state: quantum state vector (will be converted to complex64)
    :param basis: BasisBose object containing basis states and lookup map
    :return: complex matrix of shape [m, m, m, m] where m is the number of modes
    """
    return hamiltonian_bose_cy.expect_quad(
        state.astype(np.complex64),
        np.array(basis.basis, dtype=np.uint8),
        basis.lookup_map,
        basis.N
    )

def expect_six_bose(state, basis):
    """
    Compute 3-particle bosonic density matrix ρ_ijklmn = <ψ|b†_i b†_j b†_k b_l b_m b_n|ψ>

    :param state: quantum state vector (will be converted to complex64)
    :param basis: BasisBose object containing basis states and lookup map
    :return: complex matrix of shape [m, m, m, m, m, m] where m is the number of modes
    """
    return hamiltonian_bose_cy.expect_six(
        state.astype(np.complex64),
        np.array(basis.basis, dtype=np.uint8),
        basis.lookup_map,
        basis.N
    )