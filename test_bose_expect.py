"""
Test bosonic expectation value and density matrix functions.

This test verifies:
1. Correctness of expect_lin_bose and expect_quad_bose
2. Performance improvement vs legacy dict-based versions
3. Physical properties (hermiticity, trace, etc.)
"""
import numpy as np
import time
from pyqhe.basis import BasisBose
from pyqhe.expectation import expect_lin_bose, expect_quad_bose
from pyqhe.cython import legacy_bose

print("=" * 70)
print("PyQHE Bosonic Expectation Value Test")
print("=" * 70)

# Create a small test system
N = 3  # number of bosons
m = 4  # number of modes

print(f"\nTest system: N={N} bosons in m={m} modes")

# Generate basis
basis = BasisBose(N, m)
print(f"Basis size: {basis.basis.shape[0]} states")

# Create a simple test state (ground state of a harmonic oscillator-like system)
# We'll use just the first few basis states with random coefficients, then normalize
np.random.seed(42)
state_vec = np.zeros(basis.basis.shape[0], dtype=np.float64)
state_vec[:5] = np.random.randn(5)
state_vec = state_vec / np.linalg.norm(state_vec)

print("=" * 70)
print("TEST 1: 1-Particle Density Matrix (expect_lin_bose)")
print("=" * 70)

# Test 1: Linear expectation value (1-particle density matrix)
print("\nComputing ρ_ij = <ψ|b†_i b_j|ψ>...")

# New GIL-free version
t0 = time.time()
rho_lin_new = expect_lin_bose(state_vec, basis)
t_new = time.time() - t0
print(f"  New (GIL-free) time: {t_new*1000:.3f} ms")

# Legacy dict-based version
t0 = time.time()
rho_lin_legacy = legacy_bose.density_matrix(state_vec, np.array(basis.basis, dtype=np.uint8))
t_old = time.time() - t0
print(f"  Legacy (dict) time: {t_old*1000:.3f} ms")

print(f"  Speedup: {t_old/t_new:.2f}x")

# Compare results
max_diff = np.max(np.abs(rho_lin_new - rho_lin_legacy))
print(f"  Max difference: {max_diff:.2e}")

# Note: Using complex64 introduces ~1e-7 precision, so we use a reasonable tolerance
if max_diff < 1e-6:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

# Check physical properties
is_hermitian = np.allclose(rho_lin_new, rho_lin_new.T.conj())
trace = np.trace(rho_lin_new).real
print(f"\n  Hermitian: {is_hermitian}")
print(f"  Trace (should be N={N}): {trace:.6f}")

print("\n" + "=" * 70)
print("TEST 2: 2-Particle Density Matrix (expect_quad_bose)")
print("=" * 70)

# Test 2: Quadratic expectation value (2-particle density matrix)
print("\nComputing ρ_ijkl = <ψ|b†_i b†_j b_k b_l|ψ>...")

# New GIL-free version
t0 = time.time()
rho_quad_new = expect_quad_bose(state_vec, basis)
t_new = time.time() - t0
print(f"  New (GIL-free) time: {t_new*1000:.3f} ms")

# Legacy dict-based version
t0 = time.time()
rho_quad_legacy = legacy_bose.density_matrix_two(state_vec, np.array(basis.basis, dtype=np.uint8))
t_old = time.time() - t0
print(f"  Legacy (dict) time: {t_old*1000:.3f} ms")

print(f"  Speedup: {t_old/t_new:.2f}x")

# Compare results
max_diff = np.max(np.abs(rho_quad_new - rho_quad_legacy))
print(f"  Max difference: {max_diff:.2e}")

# Note: Using complex64 introduces ~1e-7 precision, so we use a reasonable tolerance
if max_diff < 1e-6:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

# Check physical properties
# For 2-particle density matrix, check a few symmetries
# ρ_ijkl should equal ρ*_klij (hermiticity)
rho_transpose = np.transpose(rho_quad_new, (3, 2, 1, 0)).conj()
is_hermitian = np.allclose(rho_quad_new, rho_transpose)
print(f"\n  Hermitian: {is_hermitian}")

print("\n" + "=" * 70)
print("TEST 3: Performance Scaling")
print("=" * 70)

print("\n  N   m    Basis  Linear (ms)  Speedup  Quad (ms)  Speedup")
print("-" * 70)

for N_test, m_test in [(3, 4), (3, 5), (4, 5), (4, 6)]:
    basis_test = BasisBose(N_test, m_test)
    Nbasis = basis_test.basis.shape[0]

    # Create random normalized state
    state_test = np.random.randn(Nbasis)
    state_test = state_test / np.linalg.norm(state_test)

    # Test linear
    t0 = time.time()
    rho_new = expect_lin_bose(state_test, basis_test)
    t_new_lin = time.time() - t0

    t0 = time.time()
    rho_old = legacy_bose.density_matrix(state_test, np.array(basis_test.basis, dtype=np.uint8))
    t_old_lin = time.time() - t0

    speedup_lin = t_old_lin / t_new_lin

    # Test quadratic (only for smaller systems)
    if Nbasis < 100:
        t0 = time.time()
        rho_new = expect_quad_bose(state_test, basis_test)
        t_new_quad = time.time() - t0

        t0 = time.time()
        rho_old = legacy_bose.density_matrix_two(state_test, np.array(basis_test.basis, dtype=np.uint8))
        t_old_quad = time.time() - t0

        speedup_quad = t_old_quad / t_new_quad

        print(f"  {N_test}   {m_test}    {Nbasis:4d}      {t_new_lin*1000:6.2f}    {speedup_lin:5.2f}x   {t_new_quad*1000:6.2f}    {speedup_quad:5.2f}x")
    else:
        print(f"  {N_test}   {m_test}    {Nbasis:4d}      {t_new_lin*1000:6.2f}    {speedup_lin:5.2f}x   (skipped)")

print("\n" + "=" * 70)
print("TEST 4: Physical Application - Particle Distribution")
print("=" * 70)

# Use the 1-particle density matrix to compute the particle distribution
# The diagonal elements ρ_ii give the expected occupation <n_i> at each site
basis_app = BasisBose(4, 6)
state_app = np.zeros(basis_app.basis.shape[0])
state_app[0] = 1.0  # Ground state (all particles in first mode)

rho_app = expect_lin_bose(state_app, basis_app)
occupations = np.diag(rho_app).real

print("\nGround state (all 4 bosons in mode 0):")
print(f"  Expected occupations: {occupations}")
print(f"  Total particles: {np.sum(occupations):.1f} (should be 4)")

if np.abs(occupations[0] - 4.0) < 1e-10 and np.sum(np.abs(occupations[1:])) < 1e-10:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

print("\n" + "=" * 70)
print("SUCCESS: All bosonic expectation value tests completed!")
print("=" * 70)
