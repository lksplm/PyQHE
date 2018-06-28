from .basis import *
from .cython.hamiltonian_cy import *

class Operator:

    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = sp.coo_matrix((10,10))
        else:
            self.matrix = matrix

    def is_hermitian(self):
        return (self.matrix.conj().T != self.matrix).nnz == 0

    @property
    def sp(self):
        return self.matrix

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dense(self):
        return self.matrix.todense()

    def __add__(self, other):
        return Operator(self.matrix + other.matrix)

    def dot(self, other):
        if isinstance(other, Operator):
            return self.matrix.dot(other.matrix)
        else:
            return self.matrix.dot(other)

    def commutator(self, other):
        return self.dot(other) - other.dot(self)

    def commutes(self, other):
        #err = (self.dot(other) != other.dot(self)).nnz does not work for really small differences on order 1e-16
        err = np.abs(self.commutator(other))
        if err.max() < 1.e-15:
            return True
        else:
            print(err)
            return False

class OperatorDiag(Operator):

    def __init__(self, basis, site_indices, spin_indices, op_func):
        V, I, J = [0.0, ], [0, ], [0, ]
        for i in range(basis.Nbasis):
            for s, t in spin_indices:
                for j, k in site_indices:
                    fac, s_out = basis.diag_op_idx(i, site_idx=j, spin_idx=s)
                    fac *= op_func(j, k)
                    if fac != 0 and s_out !=-1:
                        V.append(fac)
                        I.append(s_out)
                        J.append(i)

        self.matrix = sp.coo_matrix((V, (I, J)), shape=(basis.Nbasis, basis.Nbasis)).tocsr()

class OperatorLin(Operator):

    def __init__(self, basis, site_indices, spin_indices, op_func):
        V, I, J = [0.0, ], [0, ], [0, ]
        for i in range(basis.Nbasis):
            for s, t in spin_indices:
                for j, k in site_indices:
                    fac, s_out = basis.lin_op_idx(i, site_idx=[j,k], spin_idx=[s,t])
                    fac *= op_func(j, k)
                    if fac != 0 and s_out !=-1:
                        V.append(fac)
                        I.append(s_out)
                        J.append(i)

        self.matrix = sp.coo_matrix((V, (I, J)), shape=(basis.Nbasis, basis.Nbasis)).tocsr()

class OperatorQuad(Operator):

    def __init__(self, basis, site_indices, spin_indices, op_func):
        V, I, J = [0.0, ], [0, ], [0, ]
        for i in range(basis.Nbasis):
            for spin_i in spin_indices:
                for site_i in site_indices:
                    fac, s_out = basis.quad_op_idx(i, site_idx=site_i, spin_idx=spin_i)

                    if fac != 0 and s_out !=-1:
                        x = op_func(*site_i)
                        V.append(fac*x)
                        I.append(s_out)
                        J.append(i)

        self.matrix = sp.coo_matrix((V, (I, J)), shape=(basis.Nbasis, basis.Nbasis)).tocsr()

class OperatorLinCy(Operator):

    def __init__(self, basis, site_indices, spin_indices, op_func):
        #check array dimensions
        #site_indices = np.asarray(site_indices, np.uint32)
        #spin_indices = np.asarray(spin_indices, np.uint32)
        #assert len(site_indices.shape) == 2
        #assert site_indices.shape[1] == 2
        #assert len(spin_indices.shape) == 2
        #assert spin_indices.shape[1] == 2
        #calculate coefficients if op_func is callable, else expects array
        if callable(op_func):
            coeff = np.zeros((basis.m[0], basis.m[0], 2, 2), dtype=np.float64)
            for i, spin_i in enumerate(spin_indices):
                for j, site_i in enumerate(site_indices):
                    #print(site_i+spin_i)
                    coeff[site_i+spin_i] = op_func(*site_i, *spin_i)

        elif isinstance(op_func, np.ndarray):
            assert op_func.shape == (basis.m[0], basis.m[0], 2, 2)
            coeff = np.array(op_func, dtype=np.float64)
        else:
            coeff= np.ones((basis.m[0], basis.m[0], 2, 2), dtype=np.float64)*op_func
        self.matrix = linear(np.array(basis.basis, dtype=np.uint8), np.array(site_indices, dtype=np.uint32), \
               np.array(spin_indices, dtype=np.uint32), coeff, np.uint32(basis.m[0]))

class OperatorQuadCy(Operator):

    def __init__(self, basis, site_indices, spin_indices, op_func_site, op_func_spin):
        #check array dimensions
        #site_indices = np.asarray(site_indices, np.uint32)
        #spin_indices = np.asarray(spin_indices, np.uint32)
        #assert len(site_indices.shape) == 2
        #assert site_indices.shape[1] == 4
        #assert len(spin_indices.shape) == 2
        #assert spin_indices.shape[1] == 4
        #calculate coefficients if op_func is callable, else expects array
        if callable(op_func_site):
            coeff_site = np.zeros((basis.m[0], basis.m[0], basis.m[0], basis.m[0]), dtype=np.float64)
            for j, site_i in enumerate(site_indices):
                coeff_site[j] = op_func_site(*site_i)

        elif isinstance(op_func_site, np.ndarray):
            assert op_func_site.shape == (basis.m[0], basis.m[0], basis.m[0], basis.m[0])
            coeff_site = np.array(op_func_site, dtype=np.float64)
        else:
            coeff_site = np.ones((basis.m[0],)*4, dtype=np.float64)*op_func_site

        if callable(op_func_spin):
            coeff_spin = np.ones((2, 2, 2, 2), dtype=np.float64)
            for i, spin_i in enumerate(spin_indices):
                coeff_spin[i] = op_func_spin(*spin_i)
        elif isinstance(op_func_spin, np.ndarray):
            assert op_func_spin.shape == (2, 2, 2, 2)
            coeff_spin = np.array(op_func_spin, dtype=np.float64)
        else:
            coeff_spin = np.ones((2,)*4, dtype=np.float64)*op_func_spin

        self.matrix = quadratic(np.array(basis.basis, dtype=np.uint8), np.array(site_indices, dtype=np.uint32), \
                                np.array(spin_indices, dtype=np.uint32), coeff_site, coeff_spin, np.uint32(basis.m[0]))


class OperatorQuadDeltaCy(Operator):

    def __init__(self, basis, coeff):
        assert len(coeff.shape) == 4
        self.matrix = quadratic_delta(np.array(basis.basis, dtype=np.uint8) , coeff, np.uint32(basis.m[0]))
