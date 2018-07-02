# distutils: language=c++
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
    cdef int n
    n = state[i] + 1
    state[i] = n
    return n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int b_(data_type_t [:] state, int i) nogil:
    cdef int n
    n = state[i]
    state[i] = n - 1
    return n

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def linear(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices, np.float64_t [:,:] coeff):
    """
    Computes a Hamiltonian of the form
    sum_basis  coeff[j,k] * b^dagger_j  b_k |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param site_indices: site indices (j,k) of operators, shape [Nsiteindicies, 2]
    :param coeff: table of coefficients, indexed via (j,k), shape [m,m]
    :return: sparse Hamiltonian
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]

    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]

    cdef int i,j,k,f1,f2,idx,st0,st1

    lut = dict(zip(tuple(map(tuple, basis)), range(Nstates)))
    cdef dict cy_lut = lut

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)

    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np

    cdef vector[int] row, col
    cdef vector[double] val
    #cdef double coeff = 1.0
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
                idx = cy_lut[tuple(sp)]
                row.push_back(i)
                col.push_back(idx)
                val.push_back(coeff[st0,st1]*sqrt(f1*f2))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def quadratic(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices,\
              np.float64_t [:,:,:,:] coeff_site):
    """
    Computes a Hamiltonian of the form
    sum_basis  b^dagger_j b^dagger_k  b_l, b_l |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param site_indices: site indices (j,k,l,m) of operators, shape [Nsiteindicies, 4]
    :param coeff_site: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :return: sparse Hamiltonian
    """

    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]

    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]

    cdef int i,j,k,f1,f2,f3,f4,idx,st0,st1,st2,st3

    lut = dict(zip(tuple(map(tuple, basis)), range(Nstates)))
    cdef dict cy_lut = lut

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
                if f3>0and f4 >0:
                    idx = cy_lut[tuple(sp)]
                    row.push_back(i)
                    col.push_back(idx)
                    val.push_back(coeff_site[st0,st1,st2,st3]*sqrt(f1*f2*f3*f4))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def quadratic_delta(data_type_t [:,:] basis, np.float64_t [:,:,:,:] coeff):
    """
    Computes a Hamiltonian of the special form
    sum_basis  delta(j+k-l-m) b^dagger_j_sig1 b^dagger_k_sig2  b_l_sig2, b_l_sig1 |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param coeff: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :param L: cutoff for one spin sector
    :return:
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i,j,k,l,m,f1,f2,f3,f4,idx

    lut = dict(zip(tuple(map(tuple, basis)), range(Nstates)))
    cdef dict cy_lut = lut

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
                                idx = cy_lut[tuple(spp)]
                                row.push_back(i)
                                col.push_back(idx)
                                val.push_back(coeff[m,l,k,j]*sqrt(f1*f2*f3*f4))

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def density_matrix(np.float64_t [:] state_vec, data_type_t [:,:] basis):
    """
    Computes the density matrix \rho_ij
    :param basis: all basis states, shape [Nstates, mtotal]
    :return: \rho_ij
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i,j,k,l,m,f1,f2,f3,f4,idx

    lut = dict(zip(tuple(map(tuple, basis)), range(Nstates)))
    cdef dict cy_lut = lut

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)
    spp_np =  np.zeros(L, dtype=data_type)
    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np
    cdef data_type_t [:] spp = spp_np

    rho =  np.zeros((L,L), dtype=np.float64)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(L):
            sp[:] = state
            #check if two states j and k are occupied
            f1 = b_(sp, j)
            if f1>0 :
                for k in range(L):
                    spp[:] = sp
                    f3 = b_dagger(spp, k)
                    if f3>0:
                        #print('spp', l ,m, tuple(spp))
                        idx = cy_lut[tuple(spp)]
                        #print(i, idx)
                        rho[k,j] += sqrt(f1*f3)*state_vec[i]*state_vec[idx]

    return rho

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def density_matrix_two(np.float64_t [:] state_vec, data_type_t [:,:] basis):
    """
    Computes the density matrix \rho_ijkl
    :param basis: all basis states, shape [Nstates, mtotal]
    :return: \rho_ijkl
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]
    cdef int i,j,k,l,m,f1,f2,f3,f4,idx

    lut = dict(zip(tuple(map(tuple, basis)), range(Nstates)))
    cdef dict cy_lut = lut

    state_np =  np.zeros(L, dtype=data_type)
    sp_np =  np.zeros(L, dtype=data_type)
    spp_np =  np.zeros(L, dtype=data_type)
    cdef data_type_t [:] state = state_np
    cdef data_type_t [:] sp = sp_np
    cdef data_type_t [:] spp = spp_np

    rho =  np.zeros((L,L,L,L), dtype=np.float64)

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
                        for m in range(L):
                            spp[:] = sp
                            f3 = b_dagger(spp, l)
                            f4 = b_dagger(spp, m)
                            if f3>0 and f4 >0:
                                idx = cy_lut[tuple(spp)]
                                rho[m,l,k,j] += sqrt(f1*f2*f3*f4)*state_vec[i]*state_vec[idx]

    return rho