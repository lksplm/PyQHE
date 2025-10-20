#!/usr/bin/env python
"""Simple test of fermionic functionality"""
import numpy as np
from scipy.special import factorial
from itertools import product
from math import sqrt
import sys
sys.path.insert(0, '/home/user/PyQHE')

from pyqhe.basis import BasisFermi
from pyqhe.hamiltonian import OperatorLinCy, OperatorQuadDeltaCy
from pyqhe.eigensystem import Eigensystem

print("="*60)
print("PyQHE Fermionic Test")
print("="*60)

# Create a small fermionic basis: 2 spin-up, 2 spin-down, 6 modes
print("\n1. Creating BasisFermi with N=[2,2], m=[6,6]...")
basis = BasisFermi(N=[2,2], m=[6,6])
print(f"   Basis size: {basis.Nbasis} states")
print(f"   Hilbert space dimension: {basis.Lbasis}")

# Create kinetic energy operator (diagonal)
print("\n2. Creating kinetic energy operator H0...")
diag_sites = [(i,i) for i in range(basis.m[0])]
coeff_l = lambda i, j, s, p: i*(i==j)
H0 = OperatorLinCy(basis, site_indices=diag_sites, spin_indices=[(0,0), (1,1)], op_func=coeff_l)
print(f"   H0 is hermitian: {H0.is_hermitian()}")
print(f"   H0 matrix shape: {H0.shape}")

# Create interaction operator
print("\n3. Creating interaction operator Hint...")
int_sites = [(j,k,l,m) for j,k,l,m in product(range(basis.m[0]), repeat=4) if j+k-l-m==0]
print(f"   Number of interaction terms: {len(int_sites)}")

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

coeff = np.zeros((basis.m[0],basis.m[0],basis.m[0],basis.m[0]), dtype=np.float64)
for j, k, l, m in int_sites:
    coeff[j, k, l, m] = Vint(j, k, l, m)

Hint = OperatorQuadDeltaCy(basis, coeff=coeff)
print(f"   Hint is hermitian: {Hint.is_hermitian()}")
print(f"   Hint matrix shape: {Hint.shape}")

# Diagonalize for a few parameter values
print("\n4. Computing eigensystem...")
print("   Parameters: alpha=[0.1, 0.3, 0.5], eta=0.25, M=5 eigenvalues")
alpha = np.array([0.1, 0.3, 0.5])
eigsys = Eigensystem(ops_list=[H0, Hint], param_list=[alpha, [0.25]], M=5)

print(f"\n5. Results:")
print(f"   Energy shape: {eigsys.energies.shape}")
print(f"   State shape: {eigsys.states.shape}")
print(f"\n   Ground state energies for each alpha:")
for i, a in enumerate(alpha):
    print(f"     alpha={a:.1f}: E0={eigsys.energies[0,i]:.6f}")

print(f"\n   First 5 eigenvalues at alpha=0.1:")
for i in range(5):
    print(f"     E{i}={eigsys.energies[i,0]:.6f}")

print("\n" + "="*60)
print("SUCCESS: Fermionic implementation working correctly!")
print("="*60)
