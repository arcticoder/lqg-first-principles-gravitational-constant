#!/usr/bin/env python3
"""
Quick test of the LQG gravitational constant derivation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig

def test_basic_calculation():
    """Test basic G calculation."""
    print("Testing LQG Gravitational Constant Calculation")
    print("=" * 50)
    
    # Create minimal config
    config = GravitationalConstantConfig(
        gamma_immirzi=0.2375,
        volume_j_max=5,  # Smaller for faster computation
        include_polymer_corrections=True,
        include_volume_corrections=True,
        include_holonomy_corrections=True,  # Enable holonomy corrections
        include_higher_order_terms=True,   # Enable higher-order terms
        verbose_output=True
    )
    
    print(f"Configuration:")
    print(f"  γ = {config.gamma_immirzi}")
    print(f"  j_max = {config.volume_j_max}")
    print(f"  Polymer corrections: {config.include_polymer_corrections}")
    
    # Create calculator
    calc = GravitationalConstantCalculator(config)
    
    # Test volume calculation
    print("\nVolume Operator Test:")
    volume_contrib = calc.volume_calc.volume_contribution_to_G()
    print(f"  Volume contribution: {volume_contrib:.3e}")
    
    # Test polymer calculation
    print("\nPolymer Correction Test:")
    polymer_factor = calc.polymer_calc.polymer_correction_factor(1.0)
    print(f"  Polymer factor: {polymer_factor:.6f}")
    
    # Test basic LQG calculation
    print("\nBasic LQG Calculation:")
    import numpy as np
    HBAR = 1.054571817e-34
    C_LIGHT = 299792458
    gamma = config.gamma_immirzi
    
    G_base = gamma * HBAR * C_LIGHT / (8 * np.pi)
    print(f"  G_base = γħc/(8π) = {G_base:.6e}")
    
    # Apply volume correction
    G_corrected = G_base + volume_contrib
    print(f"  G + volume = {G_corrected:.6e}")
    
    # Apply polymer correction
    G_final = calc.polymer_calc.polymer_G_correction(G_corrected)
    print(f"  G_final = {G_final:.6e}")
    
    # Compare with experiment
    G_exp = 6.67430e-11
    relative_error = abs(G_final - G_exp) / G_exp * 100
    
    print(f"\nComparison with Experiment:")
    print(f"  G_experimental = {G_exp:.6e}")
    print(f"  G_theoretical  = {G_final:.6e}")
    print(f"  Relative error = {relative_error:.2f}%")
    
    if relative_error < 10:
        print("  ✅ Good agreement!")
    elif relative_error < 50:
        print("  ⚠️ Reasonable agreement")
    else:
        print("  ❌ Poor agreement")
    
    return G_final

if __name__ == "__main__":
    result = test_basic_calculation()
    print(f"\nFinal Result: G = {result:.6e} m³⋅kg⁻¹⋅s⁻²")
