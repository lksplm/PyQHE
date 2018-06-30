# distutils: language=c++
import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import coo_matrix
from libcpp.vector cimport vector

data_type = np.uint8
ctypedef np.uint8_t data_type_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int sign(data_type_t [:] state, np.uint32_t i) nogil:
    cdef int sign = 0
    for j in range(i):
        sign += state[j]
    return (-1)**sign

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int c_dagger(data_type_t [:] state, int i, int spin, np.uint32_t l) nogil:
    cdef int factor
    if spin==0: #spin down sector
        if state[i]==0:
            state[i]=1
            factor = sign(state,i)#(-1)**np.sum(state[0:i])
        else:
            factor=0

    elif spin==1: #spin up sector
        if state[l+i]==0:
            state[l+i]=1
            factor = sign(state,i+l)
        else:
            factor=0
    return factor

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline int c_(data_type_t [:] state, int i, int spin, np.uint32_t l) nogil:
    cdef int factor
    if spin==0: #spin down sector
        if state[i]==0:
            factor=0
        else:
            state[i]=0
            factor = sign(state,i)

    elif spin==1: #spin up sector
        if state[l+i]==0:
            factor=0
        else:
            state[l+i]=0
            factor = sign(state,i+l)
    return factor



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def linear(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices, np.uint32_t [:,:] spin_indices, np.float64_t [:,:,:,:] coeff, np.uint32_t Ldwn):
    """
    Computes a Hamiltonian of the form
    sum_basis  coeff[j,k,sig1,sig2] * b^dagger_j_sig1  b_k_sig2 |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param site_indices: site indices (j,k) of operators, shape [Nsiteindicies, 2]
    :param spin_indices: spin indices (sig1,sig2) of operators, shape [Nspinindicies, 2]
    :param coeff: table of coefficients, indexed via (j,k,sig1,sig2), shape [m,m,2,2]
    :param Ldwn: cutoff for one spin sector
    :return: sparse Hamiltonian
    """
    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]

    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]
    cdef Py_ssize_t Nspin_idx = spin_indices.shape[0]

    cdef int i,j,k,f1,f2,idx,st0,st1,sp0,sp1

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
        for j in range(Nspin_idx):
            for k in range(Nsite_idx):
                sp[:] = state
                sp0 = spin_indices[j,0]
                sp1 = spin_indices[j,1]
                st0 = site_indices[k,0]
                st1 = site_indices[k,1]
                f1 = c_(sp, st1, sp1, Ldwn)
                f2 = c_dagger(sp, st0, sp0, Ldwn)
                if f1!=0 and f2 !=0:
                    idx = cy_lut[tuple(sp)]
                    row.push_back(i)
                    col.push_back(idx)
                    val.push_back(coeff[st0,st1,sp0,sp1]*f1*f2)

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def quadratic(data_type_t [:,:] basis, np.uint32_t [:,:] site_indices, np.uint32_t [:,:] spin_indices,\
              np.float64_t [:,:,:,:] coeff_site,  np.float64_t [:,:,:,:] coeff_spin, np.uint32_t Ldwn):
    """
    Computes a Hamiltonian of the form
    sum_basis  b^dagger_j_sig1 b^dagger_k_sig2  b_l_sig3, b_l_sig3 |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param site_indices: site indices (j,k,l,m) of operators, shape [Nsiteindicies, 4]
    :param spin_indices: spin indices (sig1,sig2,sig3,sig4) of operators, shape [Nspinindicies, 4]
    :param coeff_site: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :param coeff_spin: table of coefficients for spin dof, indexed via (sig1,sig2,sig3,sig4), shape [2,2,2,2]
    :param Ldwn: cutoff for one spin sector
    :return: sparse Hamiltonian
    """

    cdef Py_ssize_t Nstates = basis.shape[0]
    cdef Py_ssize_t L = basis.shape[1]

    cdef Py_ssize_t Nsite_idx = site_indices.shape[0]
    cdef Py_ssize_t Nspin_idx = spin_indices.shape[0]

    cdef int i,j,k,f1,f2,f3,f4,idx,st0,st1,st2,st3,sp0,sp1,sp2,sp3

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
        for j in range(Nspin_idx):
            for k in range(Nsite_idx):
                sp[:] = state
                sp0 = spin_indices[j,0]
                sp1 = spin_indices[j,1]
                sp2 = spin_indices[j,2]
                sp3 = spin_indices[j,3]
                st0 = site_indices[k,0]
                st1 = site_indices[k,1]
                st2 = site_indices[k,2]
                st3 = site_indices[k,3]
                f1 = c_(sp, st3, sp3, Ldwn)
                f2 = c_(sp, st2, sp2, Ldwn)
                if f1!=0 and f2 !=0:
                    f3 = c_dagger(sp, st1, sp1, Ldwn)
                    f4 = c_dagger(sp, st0, sp0, Ldwn)
                    if f3!=0 and f4 !=0:
                        idx = cy_lut[tuple(sp)]
                        row.push_back(i)
                        col.push_back(idx)
                        val.push_back(coeff_site[st0,st1,st2,st3]*coeff_spin[sp0,sp1,sp2,sp3]*f1*f2*f3*f4)

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def quadratic_delta(data_type_t [:,:] basis, np.float64_t [:,:,:,:] coeff, np.uint32_t Ldwn):
    """
    Computes a Hamiltonian of the special form
    sum_basis  delta(j+k-l-m) b^dagger_j_sig1 b^dagger_k_sig2  b_l_sig2, b_l_sig1 |state>
    :param basis: all basis states, shape [Nstates, mtotal]
    :param coeff: table of coefficients for site dof, indexed via (j,k,l,m) shape [m,m,m,m]
    :param Ldwn: cutoff for one spin sector
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
        for sig1 in range(2):
            for sig2 in range(2):
                if sig1!=sig2:
                    for j in range(Ldwn):
                        for k in range(Ldwn):
                            sp[:] = state
                            #print('sp b',j,k, tuple(sp))
                            #check if two states j and k are occupied
                            f1 = c_(sp, j, sig1, Ldwn)
                            f2 = c_(sp, k, sig2, Ldwn)
                            if f1!=0 and f2 !=0:
                                #print('sp', j,k, tuple(sp))
                                for l in range(Ldwn):
                                    spp[:] = sp
                                    m = j+k-l
                                    if m>=0 and m<Ldwn:
                                        #print('spp b', l ,m, tuple(spp))
                                        f3 = c_dagger(spp, l, sig2, Ldwn)
                                        f4 = c_dagger(spp, m, sig1, Ldwn)

                                        if f3!=0 and f4 !=0:
                                            #print('spp', l ,m, tuple(spp))
                                            idx = cy_lut[tuple(spp)]
                                            #print(i, idx)
                                            row.push_back(i)
                                            col.push_back(idx)
                                            val.push_back(coeff[m,l,k,j]*f1*f2*f3*f4)

    return coo_matrix((val,(row,col)), dtype=np.float64, shape=(Nstates, Nstates))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def density_matrix(np.float64_t [:] state_vec, data_type_t [:,:] basis, np.int32_t sig, np.uint32_t Ldwn):
    """
    Computes the density matrix \rho_ij^(spin1, spin2)
    :param basis: all basis states, shape [Nstates, mtotal]
    :param Ldwn: cutoff for one spin sector
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

    rho =  np.zeros((Ldwn,Ldwn), dtype=np.float64)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(Ldwn):
            sp[:] = state
            #print('sp b',j,k, tuple(sp))
            #check if two states j and k are occupied
            f1 = c_(sp, j, sig, Ldwn)
            if f1!=0 :
                #print('sp', j,k, tuple(sp))
                for k in range(Ldwn):
                    spp[:] = sp
                    f3 = c_dagger(spp, k, sig, Ldwn)
                    if f3!=0:
                        #print('spp', l ,m, tuple(spp))
                        idx = cy_lut[tuple(spp)]
                        #print(i, idx)
                        rho[k,j] += f1*f3*state_vec[i]*state_vec[idx]

    return rho

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def density_matrix_two(np.float64_t [:] state_vec, data_type_t [:,:] basis, np.int32_t sig1, np.int32_t sig2, np.uint32_t Ldwn):
    """
    Computes the density matrix \rho_ijkl^(spin1, spin2)
    :param basis: all basis states, shape [Nstates, mtotal]
    :param Ldwn: cutoff for one spin sector
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

    rho =  np.zeros((Ldwn,Ldwn,Ldwn,Ldwn), dtype=np.float64)

    for i in range(Nstates):
        state[:] = basis[i,:]
        for j in range(Ldwn):
            for k in range(Ldwn):
                sp[:] = state
                #check if two states j and k are occupied
                f1 = c_(sp, j, sig1, Ldwn)
                f2 = c_(sp, k, sig2, Ldwn)
                if f1!=0 and f2 !=0:
                    for l in range(Ldwn):
                        for m in range(Ldwn):
                            spp[:] = sp
                            f3 = c_dagger(spp, l, sig2, Ldwn)
                            f4 = c_dagger(spp, m, sig1, Ldwn)
                            if f3!=0 and f4 !=0:
                                idx = cy_lut[tuple(spp)]
                                rho[m,l,k,j] += f1*f2*f3*f4*state_vec[i]*state_vec[idx]

    return rho