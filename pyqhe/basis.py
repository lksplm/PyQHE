from math import sqrt, factorial
import numpy as np
import scipy.sparse as sp
from itertools import product, combinations, combinations_with_replacement
from scipy.special import binom, factorial
from scipy.linalg import eigh, eigvalsh
from numba import jit


def multichoose(r, s):
    return int(binom(r+s-1, s))

class Basis:

    def __init__(self):
        self.basis = []
        self.Nbasis = 0
        self.m = []
        self.basis_lut = {}

    @property
    def states(self):
        return self.basis

    def print_states(self):
        for i in range(self.Nbasis):
            st = list(str(self.basis[i,:]))
            st[0] = ' '
            st[-1] = ' '
            for j in np.atleast_1d(self.m)[:-1]:
                st[2 * j] = '|'
            print(i, ". |",''.join(st),">")

    def index(self, state):
        """
        Find the index of a state vector
        :param state: state vector
        :return: state index in basis
        """
        try:
            idx = self.basis_lut[tuple(state)]
            return idx
        except:
            return -1

class BasisBose(Basis):
    def __init__(self, N, m=None):
        self.N = N
        if m is not None:
            self.m = m
        else:
            self.m = 2*self.N

        self.basis, self.basis_l =  self._generate_basis()
        self.Nbasis = self.basis.shape[0]
        self.basis_lut = dict(zip(tuple(map(tuple, self.basis)), range(self.Nbasis)))

    def _build_lut(self):
        self.basis_lut = dict(zip(tuple(map(tuple, self.basis)), range(self.Nbasis)))

    def _generate_basis(self):
        s0 = np.zeros(self.m, dtype=np.int8)
        states = np.empty((multichoose(self.m, self.N), self.m), dtype=np.int8)
        ls = np.empty(multichoose(self.m, self.N), dtype=np.uint)
        for i, idx in enumerate(combinations_with_replacement(range(self.m), self.N)):
            s = np.zeros(self.m, dtype=np.uint8)
            np.add.at(s, list(idx), 1)
            l = np.sum(s * np.arange(self.m))
            states[i, :] = s
            ls[i] = l
        idx = np.argsort(ls)
        return states[idx], ls[idx]


    def lin_op(self, state, i, j):
        """
        computes b^dagger_i b_j |state>
        :param state: state vector
        :param i: index of b^dagger
        :param j: index of b
        :return: factor, modified state vector
        """
        s_new = state.copy()
        n_j = s_new[j]
        s_new[j] = n_j - 1
        if n_j > 0:
            n_i = s_new[i]
            s_new[i] = n_i + 1

            return sqrt(n_j*(n_i+1)), s_new
        else:
            return 0, s_new

    def lin_op_idx(self, state, i, j):
        """
        computes b^dagger_i b_j |state>
        :param state: state index
        :param i: index of b^dagger
        :param j: index of b
        :return: factor, modified state index
        """
        f, new_state = self.lin_op(self.basis[state], i, j)
        return f, self.index(new_state)

    def quad_op(self, state, j, k, l, m):
        """
        computes b^dagger_j b^dagger_k b_l b_m |state>
        :param state: state vector
        :param j: index of b^dagger
        :param k: index of b^dagger
        :param l: index of b
        :param m: index of b
        :return: factor, modified state vector
        """
        s_new = state.copy()
        n_m = s_new[m]
        s_new[m] = n_m - 1
        n_l = s_new[l]
        s_new[l] = n_l - 1
        if n_m > 0 and n_l > 0:
            n_k = s_new[k]
            s_new[k] += 1
            n_j = s_new[j]
            s_new[j] += 1

            return sqrt(n_m * n_l * (n_k + 1) * (n_j + 1)), s_new
        else:
            return 0, s_new

    def quad_op_idx(self, state, j, k, l, m):
        """
        computes b^dagger_j b^dagger_k b_l b_m |state>
        :param state: state index
        :param j: index of b^dagger
        :param k: index of b^dagger
        :param l: index of b
        :param m: index of b
        :return: factor, modified state index
        """
        f, new_state = self.quad_op(self.basis[state], j, k, l, m)
        return f, self.index(new_state)

class BasisFermi(Basis):
    def __init__(self, N=[2,2], m=None, spin_conserved=True):
        self.N_spin_comp = len(N)
        self.N = N
        if m is not None:
            self.m = np.array(m)
        else:
            self.m = np.array([2*n for n in self.N])

        if spin_conserved:
            self.basis, self.basis_l = self._build_basis(self.N, self.m)
        else:
            self.basis, self.basis_l = self._build_basis_snc(self.N, self.m)

        self.Nbasis = self.basis.shape[0]
        self._build_lut()

    def _build_basis(self, N, ms):
        """
        Generate Fermionic Basis for multiple (spin) components where the spin (i.e. particles per component) is conserved
        :param N: list [N0, N1,...] particles per component
        :param ms: list [m0, m1,...] angular momentum cutoff per component
        :return: basis array [Nstates,Mtotal], angular momenta of all states [Nstates]
        """
        states_spin = [combinations(range(m), n) for n, m in zip(N, ms)] #generate all possible indices for particles per component, combinations ensures no double occupacies
        Nstates_total = int(np.prod([int(binom(m, n)) for n, m in zip(N, ms)])) #compute total number of states as binomoial coefficient
        m_total = int(np.sum(ms))
        basis = np.zeros((Nstates_total, m_total), dtype=np.int8) #set up basis array
        st_fermi = product(*states_spin) #cartesian product between all components generates all states, still indices
        basis_l = np.zeros(Nstates_total, dtype=np.int) #angular momentum "operator" (diagonal in this basis)/index per component
        self.offs_arr = np.insert(ms[:-1], 0, 0) #compute offsets for each component for indexing
        self.offs_arr_end = np.insert(ms[1:], -1, -1)
        l_diag = np.concatenate([np.arange(m) for m in ms])
        for i, idx in enumerate(st_fermi):
            s = np.zeros(m_total, dtype=np.uint8)
            for j, m in enumerate(ms):
                if idx[j]: #ensure that list of indices is not empty
                    np.add.at(s, np.array(idx[j]) + self.offs_arr[j], 1) #create vector of zeros and add ones at particle indices
            basis[i, :] = s #insert into basis array
            basis_l[i] = np.sum(s * l_diag) #compute L for this state

        idx = np.argsort(basis_l) #sort all states wiht increasing L
        return basis[idx], basis_l[idx]

        
    def _build_basis_snc(self,N, ms):
        """
        generate basis when Spin is not conserved
        Only total number of particles is constant, distribute this number among all components and concatenate the resulting bases

        Test implementation, only for two components
        :return:
        """
        Ntotal = sum(N)
        bases = []
        bases_ls = []
        for i in range(Ntotal+1):
            #print("i={:d}, N=[{:d},{:d}]".format(i,Ntotal-i,i))
            b, l = self._build_basis([Ntotal-i,i],ms)
            bases.append(b)
            bases_ls.append(l)
        bases = np.concatenate(bases, axis=0)
        bases_ls = np.concatenate(bases_ls, axis=0)
        idx = np.argsort(bases_ls)  # sort all states wiht increasing L
        return bases[idx], bases_ls[idx]
        #return bases, bases_ls


    def _build_lut(self):
        self.basis_lut = dict(zip(tuple(map(tuple, self.basis)), range(self.Nbasis)))

    def _c_dagger(self, state, i, spin):
        j = self.offs_arr[spin]+i
        if state[j] == 0:
            state[j] = 1
            factor = (-1)**(state[0:j].sum())
        else:
            factor = 0

        return state, factor

    def _c(self, state, i, spin):
        j = self.offs_arr[spin] + i
        if state[j] == 0:
            factor = 0
        else:
            state[j] = 0
            factor = (-1) ** (state[0:j].sum())

        return state, factor

    def diag_op(self, state, site_idx, spin_idx):
        """
        computes b^dagger_j  b_k |state>
        :param state: state vector
        :param site_idx: indices in the order j,k
        :param spin_idx: spin indices in the same order
        :return: factor, modified state vector
        """
        j = self.offs_arr[spin_idx] + site_idx
        n = state[j]
        return n, state

    def diag_op_idx(self, state, site_idx, spin_idx):
        """
        computes b^dagger_i b_j |state>
        :param state: state index
        :param i: index of b^dagger
        :param j: index of b
        :return: factor, modified state index
        """
        f, new_state = self.diag_op(self.basis[state], site_idx, spin_idx)
        return f, state

    def lin_op(self, state, site_idx, spin_idx):
        """
        computes b^dagger_j  b_k |state>
        :param state: state vector
        :param site_idx: indices in the order j,k
        :param spin_idx: spin indices in the same order
        :return: factor, modified state vector
        """
        s = state.copy()

        s, f1 = self._c(s, site_idx[1], spin_idx[1])
        if f1 == 0: return 0, s
        s, f2 = self._c_dagger(s, site_idx[0], spin_idx[0])
        if f2 == 0: return 0, s

        return f1 * f2, s

    def lin_op_idx(self, state, site_idx, spin_idx):
        """
        computes b^dagger_i b_j |state>
        :param state: state index
        :param i: index of b^dagger
        :param j: index of b
        :return: factor, modified state index
        """
        f, new_state = self.lin_op(self.basis[state], site_idx, spin_idx)
        return f, self.index(new_state)

    def quad_op(self, state, site_idx, spin_idx):
        """
        computes b^dagger_j b^dagger_k b_l b_m |state>
        :param state: state vector
        :param site_idx: indices in the order j,k,l,m
        :param spin_idx: spin indices in the same order
        :return: factor, modified state vector
        """
        s = state.copy()

        s, f1 = self._c(s, site_idx[3], spin_idx[3])
        if f1 == 0: return 0, s
        s, f2 = self._c(s, site_idx[2], spin_idx[2])
        if f2 == 0: return 0, s
        s, f3 = self._c_dagger(s, site_idx[1], spin_idx[1])
        if f3 == 0: return 0, s
        s, f4 = self._c_dagger(s, site_idx[0], spin_idx[0])
        if f4 == 0: return 0, s

        return f1 * f2 * f3 * f4, s

    def quad_op_idx(self, state,  site_idx, spin_idx):
        """
        computes b^dagger_j b^dagger_k b_l b_m |state>
        :param state: state index
        :param site_idx: indices in the order j,k,l,m
        :param spin_idx: spin indices in the same order
        :return: factor, modified state index
        """
        f, new_state = self.quad_op(self.basis[state], site_idx, spin_idx)
        return f, self.index(new_state)
