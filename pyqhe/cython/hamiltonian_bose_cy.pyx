# distutils: language=c++
"""
Optimized bosonic operators using GIL-free array-based state lookup.

All functions use array-based lookup tables for fast state indexing without
acquiring the Python GIL. Legacy dict-based versions are in legacy_bose.pyx.
"""
import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import coo_matrix
from libcpp.vector cimport vector
from libc.math cimport sqrt
data_type = np.uint8
ctypedef np.uint8_t data_type_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int b_dagger(data_type_t [:] state, int i) nogil:
    """Bosonic creation operator: b†_i |state>"""
    cdef int n
    n = state[i] + 1
    state[i] = n
    return n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int b_(data_type_t [:] state, int i) nogil:
    """Bosonic annihilation operator: b_i |state>"""
    cdef int n
    n = state[i]
    state[i] = n - 1
    return n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline long long state_to_int_bose(data_type_t [:] state, Py_ssize_t L, int max_occ) nogil:
    """
    Encode a bosonic state as a unique integer using base-(max_occ+1) encoding.

    For a state with occupation numbers [n_0, n_1, ..., n_{L-1}] where each n_i <= max_occ,
    we encode it as: n_0 + n_1*B + n_2*B^2 + ... + n_{L-1}*B^{L-1}
    where B = max_occ + 1 is the base.

    This encoding is GIL-free and provides O(1) lookup when used with a pre-built
    array-based lookup table.

    :param state: bosonic state occupation number vector
    :param L: number of modes
    :param max_occ: maximum occupation number (typically N, the total number of particles)
    :return: unique integer encoding of the state
    """
    cdef long long out = 0
    cdef long long base_power = 1
    cdef int base = max_occ + 1
    cdef Py_ssize_t j

    for j in range(L):
        out += state[j] * base_power
        base_power *= base

    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def linear(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices,
           np.float64_t [:,:] coeff, np.int32_t [:] lut, int max_occ):
    """
    Computes a Hamiltonian of the form
    sum_basis  coeff[j,k] * b^dagger_j  b_k |state>

    Uses GIL-free array-based lookup for performance.

    :param basis: all basis states, shape [Nstates, m]
    :param site_indices: site indices (j,k) of operators, shape [Nsiteindicies, 2]
    :param coeff: table of coefficients, indexed via (j,k), shape [m,m]
    :param lut: lookup table mapping state_to_int -> basis index
    :param max_occ: maximum occupation number (typically N)
    :return: sparse Hamiltonian
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]
    cdef int i,j,k,f1,f2,idx,st0,st1
    cdef long long state_int

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)

    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np

    cdef vector[int] row, col
    cdef vector[double] val
    row.reserve(L**2)
    col.reserve(L**2)
    val.reserve(L**2)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for k in range(Nsite_idx):
            sp[:] = state
            st0 = site_indices[k,0]
            st1 = site_indices[k,1]
            f1 = b_(sp, st1)
            f2 = b_dagger(sp, st0)
            if f1>0 and f2 >0:
                state_int = state_to_int_bose(sp, L, max_occ)
                idx = lut[state_int]
                if idx > -1:
                    row.push_back(i)
                    col.push_back(idx)
                    val.push_back(coeff[st0,st1]*sqrt(f1*f2))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False)
@cython.wraparound(False)
def quadratic(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices,
              np.float64_t [:,:,:,:] coeff_site, np.int32_t [:] lut, int max_occ):
    """
    Computes a Hamiltonian of the form
    sum_basis  b^dagger_j b^dagger_k  b_l, b_m |state>

    Uses GIL-free array-based lookup for performance.

    :param basis: all basis states, shape [Nstates, m]
    :param site_indices: site indices (j,k,l,m) of operators, shape [Nsiteindicies, 4]
    :param coeff_site: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :param lut: lookup table mapping state_to_int -> basis index
    :param max_occ: maximum occupation number (typically N)
    :return: sparse Hamiltonian
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]
    cdef int i,j,k,f1,f2,f3,f4,idx,st0,st1,st2,st3
    cdef long long state_int

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)

    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np

    cdef vector[int] row, col
    cdef vector[double] val

    row.reserve(L*Nstates)
    col.reserve(L*Nstates)
    val.reserve(L*Nstates)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for k in range(Nsite_idx):
            sp[:] = state
            st0 = site_indices[k,0]
            st1 = site_indices[k,1]
            st2 = site_indices[k,2]
            st3 = site_indices[k,3]
            f1 = b_(sp, st3)
            f2 = b_(sp, st2)
            if f1>0 and f2 >0:
                f3 = b_dagger(sp, st1)
                f4 = b_dagger(sp, st0)
                if f3>0 and f4 >0:
                    state_int = state_to_int_bose(sp, L, max_occ)
                    idx = lut[state_int]
                    if idx > -1:
                        row.push_back(i)
                        col.push_back(idx)
                        val.push_back(coeff_site[st0,st1,st2,st3]*sqrt(f1*f2*f3*f4))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False)
@cython.wraparound(False)
def quadratic_delta(data_type_t [:,:] basis, np.float64_t [:,:,:,:] coeff,
                    np.int32_t [:] lut, int max_occ):
    """
    Computes a Hamiltonian of the special form
    sum_basis  delta(j+k-l-m) b^dagger_j b^dagger_k  b_l b_m |state>

    Uses GIL-free array-based lookup for performance.

    :param basis: all basis states, shape [Nstates, m]
    :param coeff: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :param lut: lookup table mapping state_to_int -> basis index
    :param max_occ: maximum occupation number (typically N)
    :return: sparse Hamiltonian
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i,j,k,l,m,f1,f2,f3,f4,idx
    cdef long long state_int

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)
    spp_np =  np.zeros(L, dtype=data_type)
    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np
    cdef data_type_t [:] spp = spp_np

    cdef vector[int] row, col
    cdef vector[double] val

    row.reserve(L*Nstates)
    col.reserve(L*Nstates)
    val.reserve(L*Nstates)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(L):
            for k in range(L):
                sp[:] = state
                #check if two states j and k are occupied
                f1 = b_(sp, j)
                f2 = b_(sp, k)
                if f1>0 and f2>0:
                    for l in range(L):
                        spp[:] = sp
                        m = j+k-l
                        if m>=0 and m<L:
                            f3 = b_dagger(spp, l)
                            f4 = b_dagger(spp, m)

                            if f3>0 and f4>0:
                                state_int = state_to_int_bose(spp, L, max_occ)
                                idx = lut[state_int]
                                if idx > -1:
                                    row.push_back(i)
                                    col.push_back(idx)
                                    val.push_back(coeff[m,l,k,j]*sqrt(f1*f2*f3*f4))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False)
@cython.wraparound(False)
def expect_lin(np.complex64_t [:] state_vec, data_type_t [:,:] basis,
               np.int32_t [:] lut, int max_occ):
    """
    Computes the 1-particle bosonic density matrix ρ_ij = <ψ|b†_i b_j|ψ>

    Uses GIL-free array-based lookup for performance.

    :param state_vec: quantum state vector, shape [Nstates]
    :param basis: all basis states, shape [Nstates, L]
    :param lut: array-based lookup map from state integer to basis index
    :param max_occ: maximum occupation per mode (N)
    :return: ρ_ij, shape [L, L]
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i, j, k, idx
    cdef int f1, f2
    cdef long long state_int

    state_np = np.zeros(L, dtype=data_type)
    sp_np = np.zeros(L, dtype=data_type)
    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np

    cdef np.complex64_t [:,:] rho = np.zeros((L, L), dtype=np.complex64)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(L):
            sp[:] = state
            f1 = b_(sp, j)
            if f1 > 0:
                for k in range(L):
                    f2 = b_dagger(sp, k)
                    if f2 > 0:
                        state_int = state_to_int_bose(sp, L, max_occ)
                        idx = lut[state_int]
                        if idx > -1:
                            rho[k, j] += sqrt(f1 * f2) * np.conj(state_vec[i]) * state_vec[idx]
                    sp[:] = state  # reset for next k
                    sp[j] -= 1  # reapply b_j

    return np.asarray(rho)

@cython.boundscheck(False)
@cython.wraparound(False)
def expect_quad(np.complex64_t [:] state_vec, data_type_t [:,:] basis,
                np.int32_t [:] lut, int max_occ):
    """
    Computes the 2-particle bosonic density matrix ρ_ijkl = <ψ|b†_i b†_j b_k b_l|ψ>

    Uses GIL-free array-based lookup for performance.

    :param state_vec: quantum state vector, shape [Nstates]
    :param basis: all basis states, shape [Nstates, L]
    :param lut: array-based lookup map from state integer to basis index
    :param max_occ: maximum occupation per mode (N)
    :return: ρ_ijkl, shape [L, L, L, L]
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i, j, k, l, m, idx
    cdef int f1, f2, f3, f4
    cdef long long state_int

    state_np = np.zeros(L, dtype=data_type)
    sp_np = np.zeros(L, dtype=data_type)
    spp_np = np.zeros(L, dtype=data_type)
    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np
    cdef data_type_t [:] spp = spp_np

    cdef np.complex64_t [:,:,:,:] rho = np.zeros((L, L, L, L), dtype=np.complex64)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(L):
            for k in range(L):
                sp[:] = state
                f1 = b_(sp, j)
                f2 = b_(sp, k)
                if f1 > 0 and f2 > 0:
                    for l in range(L):
                        for m in range(L):
                            spp[:] = sp
                            f3 = b_dagger(spp, l)
                            f4 = b_dagger(spp, m)
                            if f3 > 0 and f4 > 0:
                                state_int = state_to_int_bose(spp, L, max_occ)
                                idx = lut[state_int]
                                if idx > -1:
                                    rho[m, l, k, j] += sqrt(f1*f2*f3*f4) * np.conj(state_vec[i]) * state_vec[idx]

    return np.asarray(rho)
