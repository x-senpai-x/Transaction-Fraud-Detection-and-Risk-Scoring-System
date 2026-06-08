import numpy as np

from transaction_risk_engine.monitoring.drift import calculate_psi


def test_calculate_psi_identical():
    # Identical distributions should have PSI 0
    ref = np.random.normal(0, 1, 1000)
    cur = ref.copy()
    
    psi = calculate_psi(ref, cur, buckets=10)
    # Numerical instability might make it slightly above 0, but very small
    assert psi < 1e-5


def test_calculate_psi_different():
    # Different distributions should have PSI > 0.1
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(2, 1, 1000)
    
    psi = calculate_psi(ref, cur, buckets=10)
    assert psi > 0.5


def test_calculate_psi_zero_handling():
    # Ensure it doesn't crash on completely separate distributions causing 0 counts in bins
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cur = np.array([0.9, 0.95, 0.96, 0.97, 0.99])
    
    psi = calculate_psi(ref, cur, buckets=5)
    # Should not throw exception or return NaN
    assert np.isfinite(psi)
    assert psi > 0
