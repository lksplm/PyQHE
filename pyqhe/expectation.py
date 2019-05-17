from .basis import *
from .cython.hamiltonian_cy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def expect_lin(state, basis, a, b):
    return expectation_lin(state.astype(np.complex64), basis.states, np.int32(a), np.int32(b), np.int32(basis.Lbasis//2))

def expect_quad(state, basis, a, b, c, d):
    return expectation_quad_test(state.astype(np.complex64), basis.states, basis.lookup_map, np.int32(a), np.int32(b),\
                                 np.int32(c), np.int32(d), np.int32(basis.Lbasis//2))

def expect_six(state, basis, a, b, c, d, e, f):
    return expectation_six_test(state.astype(np.complex64), basis.states, basis.lookup_map, np.int32(a), np.int32(b),\
                                 np.int32(c), np.int32(d), np.int32(e), np.int32(f), np.int32(basis.Lbasis//2))