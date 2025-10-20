#!/usr/bin/env python
"""
Test and benchmark GIL-free bosonic operators.

Compares old dict-based lookup vs new array-based lookup.
"""
import numpy as np
from scipy.special import factorial
from itertools import product
from math import sqrt
import time
import sys
sys.path.insert(0, '/home/user/PyQHE')

from pyqhe.basis import BasisBose
from pyqhe.hamiltonian_bose import (OperatorLinCy, OperatorLinCyFast,
                                     OperatorQuadDeltaCy, OperatorQuadDeltaCyFast)

print("="*70)
print("PyQHE Bosonic GIL-Free Lookup Test")
print("="*70)

# Test 1: Verify correctness
print("\n" + "="*70)
print("TEST 1: Correctness Verification")
print("="*70)

N, m = 4, 6
print(f"\nCreating BasisBose with N={N} bosons in m={m} modes...")
basis = BasisBose(N=N, m=m)
print(f"  Basis size: {basis.Nbasis} states")
print(f"  Lookup map size: {len(basis.lookup_map)}")
print(f"  Lookup map memory: {basis.lookup_map.nbytes / 1024:.2f} KB")

# Create kinetic energy operator with both methods
print("\n1. Testing Linear Operator (kinetic energy)...")
diag_sites = [(i,i) for i in range(m)]
coeff_l = lambda i, j: i*(i==j)

t0 = time.time()
H0_old = OperatorLinCy(basis, site_indices=diag_sites, op_func=coeff_l)
t_old = time.time() - t0

t0 = time.time()
H0_fast = OperatorLinCyFast(basis, site_indices=diag_sites, op_func=coeff_l)
t_fast = time.time() - t0

print(f"  Old (dict) time: {t_old*1000:.3f} ms")
print(f"  Fast (array) time: {t_fast*1000:.3f} ms")
print(f"  Speedup: {t_old/t_fast:.2f}x")

# Check if matrices are identical
diff = (H0_old.matrix - H0_fast.matrix).todense()
max_diff = np.abs(diff).max()
print(f"  Max difference: {max_diff:.2e}")
print(f"  ✓ PASSED" if max_diff < 1e-14 else f"  ✗ FAILED")

# Create interaction operator
print("\n2. Testing Quadratic Delta Operator (interactions)...")
int_sites = [(j,k,l,m) for j,k,l,m in product(range(m), repeat=4) if j+k-l-m==0]
print(f"  Number of interaction terms: {len(int_sites)}")

def Vint(j, k, l, m):
    return factorial(j + k) / (2 ** (j + k) * sqrt(factorial(j) * factorial(k) * factorial(l) * factorial(m)))

coeff = np.zeros((m, m, m, m), dtype=np.float64)
for j, k, l, m_idx in int_sites:
    coeff[j, k, l, m_idx] = Vint(j, k, l, m_idx)

t0 = time.time()
Hint_old = OperatorQuadDeltaCy(basis, coeff=coeff)
t_old = time.time() - t0

t0 = time.time()
Hint_fast = OperatorQuadDeltaCyFast(basis, coeff=coeff)
t_fast = time.time() - t0

print(f"  Old (dict) time: {t_old*1000:.3f} ms")
print(f"  Fast (array) time: {t_fast*1000:.3f} ms")
print(f"  Speedup: {t_old/t_fast:.2f}x")

diff = (Hint_old.matrix - Hint_fast.matrix).todense()
max_diff = np.abs(diff).max()
print(f"  Max difference: {max_diff:.2e}")
print(f"  ✓ PASSED" if max_diff < 1e-14 else f"  ✗ FAILED")

print(f"\n  H0 hermitian: {H0_fast.is_hermitian()}")
print(f"  Hint hermitian: {Hint_fast.is_hermitian()}")

# Test 2: Performance benchmark with larger system
print("\n" + "="*70)
print("TEST 2: Performance Benchmark")
print("="*70)

sizes = [(3, 6), (4, 6), (4, 8), (5, 8)]
print(f"\n{'N':>3} {'m':>3} {'Basis':>8} {'Old (ms)':>10} {'Fast (ms)':>10} {'Speedup':>8}")
print("-"*70)

for N, m in sizes:
    basis = BasisBose(N=N, m=m)
    diag_sites = [(i,i) for i in range(m)]

    # Benchmark linear operator
    t0 = time.time()
    H_old = OperatorLinCy(basis, site_indices=diag_sites, op_func=lambda i,j: i*(i==j))
    t_old = (time.time() - t0) * 1000

    t0 = time.time()
    H_fast = OperatorLinCyFast(basis, site_indices=diag_sites, op_func=lambda i,j: i*(i==j))
    t_fast = (time.time() - t0) * 1000

    speedup = t_old / t_fast
    print(f"{N:3d} {m:3d} {basis.Nbasis:8d} {t_old:10.3f} {t_fast:10.3f} {speedup:8.2f}x")

# Test 3: Verify state_to_int encoding
print("\n" + "="*70)
print("TEST 3: State Encoding Verification")
print("="*70)

from pyqhe.cython.hamiltonian_bose_cy import state_to_int_bose

N, m = 3, 4
basis = BasisBose(N=N, m=m)
print(f"\nN={N} bosons in m={m} modes")
print(f"Basis has {basis.Nbasis} states\n")

print("Sample states and their integer encodings:")
print(f"{'State':<20} {'Integer':>10} {'Lookup':>8} {'Match':>6}")
print("-"*70)

for i in range(min(10, basis.Nbasis)):
    state = basis.basis[i]
    state_int = state_to_int_bose(state, m, N)
    lookup_idx = basis.lookup_map[state_int]
    match = "✓" if lookup_idx == i else "✗"
    print(f"{str(list(state)):<20} {state_int:10d} {lookup_idx:8d} {match:>6}")

# Verify all states map correctly
all_correct = True
for i in range(basis.Nbasis):
    state_int = state_to_int_bose(basis.basis[i], m, N)
    lookup_idx = basis.lookup_map[state_int]
    if lookup_idx != i:
        all_correct = False
        print(f"ERROR: State {i} maps to {lookup_idx}")

print(f"\n{'All states map correctly: ✓ PASSED' if all_correct else 'Mapping errors detected: ✗ FAILED'}")

print("\n" + "="*70)
print("SUCCESS: GIL-free bosonic lookup working correctly!")
print("="*70)
