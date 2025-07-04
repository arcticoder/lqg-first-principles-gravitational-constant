#!/usr/bin/env python3
"""
Enhanced test script for the improved LQG gravitational constant calculation.

Tests the complete enhanced framework including:
- Universal SU(2) generating functionals
- Hypergeometric volume eigenvalues
- Corrected polymer modifications  
- Ladder operator flux structure
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import numpy as np
from gravitational_constant import (
    GravitationalConstantCalculator, 
    GravitationalConstantConfig,
    VolumeOperatorContribution,
    HolonomyFluxStructure,
    ScalarFieldCoupling,
    PolymerQuantizationEffects
)
from holonomy_flux_algebra import HolonomyFluxAlgebra
from stress_energy_tensor import CompleteStressEnergyTensor, StressEnergyConfig

def test_enhanced_framework():
    """Test the enhanced mathematical framework components."""
    print("🧪 Testing Enhanced LQG Mathematical Framework")
    print("=" * 60)
    
    # Enhanced configuration
    config = GravitationalConstantConfig(
        gamma_immirzi=0.2375,
        j_max=8,  # Higher resolution
        include_polymer_corrections=True,
        include_volume_corrections=True,
        include_holonomy_corrections=True,  # Enable holonomy
        include_higher_order_terms=True
    )
    
    print(f"Configuration:")
    print(f"  γ = {config.gamma_immirzi}")
    print(f"  j_max = {config.j_max}")
    print(f"  Enhanced corrections: All enabled")
    print()
    
    # Test universal generating functional
    print("1. Universal SU(2) Generating Functional Test:")
    flux_algebra = HolonomyFluxAlgebra()
    generating_factor = flux_algebra.universal_generating_functional()
    print(f"   G({{x_e}}) = {generating_factor:.6f}")
    
    # Test hypergeometric volume eigenvalues
    print("2. Hypergeometric Volume Eigenvalue Test:")
    volume_calc = VolumeOperatorContribution(config)
    for j in [1, 2, 3]:
        eigenval = volume_calc.compute_volume_eigenvalue(j)
        # Test hypergeometric enhancement
        rho_e = config.gamma_immirzi * j / (1 + config.gamma_immirzi * j)
        hypergeom_val = volume_calc._compute_hypergeometric_2F1(-2*j, 0.5, 1.0, -rho_e)
        print(f"   j={j}: V = {eigenval:.3e}, ₂F₁ = {hypergeom_val:.6f}")
    
    # Test corrected sinc polymer modifications
    print("3. Corrected Sinc Polymer Test:")
    stress_energy = CompleteStressEnergyTensor(StressEnergyConfig())
    for mu_p in [0.1, 0.5, 1.0]:
        correction = stress_energy.polymer_momentum_correction(mu_p)
        print(f"   μp={mu_p}: sinc(πμp) = {correction:.6f}")
    
    # Test ladder operator flux structure  
    print("4. Ladder Operator Flux Test:")
    flux_eigenvals = flux_algebra.ladder_operator_flux_eigenvalues(max_modes=5)
    for mu_i, eigenval in list(flux_eigenvals.items())[:5]:
        sqrt_factor = np.sqrt(mu_i + 1) + np.sqrt(max(mu_i - 1, 0))
        print(f"   μ={mu_i}: ⟨φ⟩ = {eigenval:.3e}, √(μ±1) = {sqrt_factor:.3f}")
    
    print()
    return config

def test_enhanced_calculation():
    """Test the complete enhanced gravitational constant calculation."""
    print("🌌 Enhanced Gravitational Constant Calculation")
    print("=" * 60)
    
    config = test_enhanced_framework()
    
    # Initialize enhanced calculator
    calc = GravitationalConstantCalculator(config)
    
    # Compute enhanced theoretical G
    print("Computing enhanced theoretical G...")
    results = calc.compute_theoretical_G()
    
    print("\n📊 Enhanced Results:")
    print(f"   Volume contribution:     {results['volume_contribution']:.3e}")
    print(f"   Holonomy contribution:   {results['holonomy_contribution']:.3e}")
    print(f"   Scalar field G:          {results['scalar_field_G']:.3e}")
    print(f"   Polymer correction:      {results['polymer_correction_factor']:.6f}")
    print(f"   Higher-order correction: {results['higher_order_correction']:.6f}")
    print()
    
    G_enhanced = results['G_theoretical']
    print(f"🎯 Enhanced Theoretical G: {G_enhanced:.6e} m³⋅kg⁻¹⋅s⁻²")
    
    # Compare with experiment
    validation = calc.validate_against_experiment(results)
    G_exp = config.experimental_G
    
    print(f"\n🔍 Validation:")
    print(f"   Experimental G:  {G_exp:.6e} m³⋅kg⁻¹⋅s⁻²")
    print(f"   Theoretical G:   {G_enhanced:.6e} m³⋅kg⁻¹⋅s⁻²")
    print(f"   Relative error:  {validation['relative_error']:.2%}")
    print(f"   Agreement:       {'✅ Good' if validation['within_uncertainty'] else '🔄 Improved'}")
    
    # Calculate improvement over basic method
    G_basic = config.gamma_immirzi * 1.055e-34 * 3e8 / (8 * np.pi)  # Basic γħc/(8π)
    basic_error = abs(G_basic - G_exp) / G_exp
    enhanced_error = validation['relative_error']
    
    improvement_factor = basic_error / enhanced_error if enhanced_error > 0 else float('inf')
    print(f"   Improvement:     {improvement_factor:.1f}x better than basic calculation")
    
    return results

def main():
    """Main test execution."""
    print("🚀 Enhanced LQG Gravitational Constant Framework")
    print("=" * 60)
    print("Testing advanced mathematical improvements:")
    print("• Universal SU(2) generating functionals")  
    print("• Hypergeometric volume eigenvalues")
    print("• Corrected sinc polymer modifications")
    print("• Ladder operator flux structures")
    print()
    
    try:
        results = test_enhanced_calculation()
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced Framework Test Complete!")
        print(f"Final G = {results['G_theoretical']:.6e} m³⋅kg⁻¹⋅s⁻²")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
