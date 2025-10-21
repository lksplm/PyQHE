from .basis import *
from .cython.hamiltonian_bose_cy import *
from .hamiltonian import Operator

class OperatorLinCy(Operator):
    """Linear bosonic operator with GIL-free array-based lookup."""

    def __init__(self, basis, site_indices, op_func):
        # Calculate coefficients if op_func is callable, else expects array
        if callable(op_func):
            coeff = np.zeros((basis.m, basis.m), dtype=np.float64)
            for j, site_i in enumerate(site_indices):
                coeff[site_i] = op_func(*site_i)
        elif isinstance(op_func, np.ndarray):
            assert op_func.shape == (basis.m, basis.m)
            coeff = np.array(op_func, dtype=np.float64)
        else:
            coeff = np.ones((basis.m, basis.m), dtype=np.float64)*op_func

        self.matrix = linear(np.array(basis.basis, dtype=np.uint8),
                            np.array(site_indices, dtype=np.uint32),
                            coeff,
                            basis.lookup_map,
                            basis.N)

class OperatorQuadCy(Operator):
    """Quadratic bosonic operator with GIL-free array-based lookup."""

    def __init__(self, basis, site_indices, spin_indices, op_func_site, op_func_spin):
        # Note: spin_indices and op_func_spin are ignored for bosons (no spin)
        # Calculate coefficients if op_func is callable, else expects array
        if callable(op_func_site):
            coeff_site = np.zeros((basis.m, basis.m, basis.m, basis.m), dtype=np.float64)
            for j, site_i in enumerate(site_indices):
                coeff_site[site_i] = op_func_site(*site_i)
        elif isinstance(op_func_site, np.ndarray):
            assert op_func_site.shape == (basis.m, basis.m, basis.m, basis.m)
            coeff_site = np.array(op_func_site, dtype=np.float64)
        else:
            coeff_site = np.ones((basis.m,)*4, dtype=np.float64)*op_func_site

        self.matrix = quadratic(np.array(basis.basis, dtype=np.uint8),
                               np.array(site_indices, dtype=np.uint32),
                               coeff_site,
                               basis.lookup_map,
                               basis.N)

class OperatorQuadDeltaCy(Operator):
    """Delta-constrained quadratic bosonic operator with GIL-free array-based lookup."""

    def __init__(self, basis, coeff):
        assert len(coeff.shape) == 4
        self.matrix = quadratic_delta(np.array(basis.basis, dtype=np.uint8),
                                     coeff,
                                     basis.lookup_map,
                                     basis.N)

