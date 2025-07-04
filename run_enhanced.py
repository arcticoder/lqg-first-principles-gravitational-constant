#!/usr/bin/env python3
"""
Run the complete enhanced LQG gravitational constant calculation.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig

def main():
    print("🌌 Enhanced LQG Gravitational Constant Calculation")
    print("=" * 60)
    
    # Enhanced configuration with all corrections enabled
    config = GravitationalConstantConfig(
        gamma_immirzi=0.2375,
        volume_j_max=8,  # Higher resolution for enhanced accuracy
        include_polymer_corrections=True,
        include_volume_corrections=True,
        include_holonomy_corrections=True,
        include_higher_order_terms=True,
        verbose_output=True
    )
    
    print(f"Enhanced Configuration:")
    print(f"  γ = {config.gamma_immirzi}")
    print(f"  j_max = {config.volume_j_max}")
    print(f"  All corrections enabled")
    print()
    
    # Create enhanced calculator
    calc = GravitationalConstantCalculator(config)
    
    # Run complete theoretical calculation
    print("Computing complete theoretical G with all enhancements...")
    results = calc.compute_theoretical_G()
    
    print("\n📊 Enhanced Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if 'G' in key or 'contribution' in key:
                print(f"   {key:25}: {value:.6e}")
            else:
                print(f"   {key:25}: {value:.6f}")
    
    # Validation against experiment
    validation = calc.validate_against_experiment(results)
    
    print(f"\n🔍 Validation Results:")
    G_exp = config.experimental_G
    G_theory = results['G_theoretical']
    
    print(f"   Experimental G:  {G_exp:.6e} m³⋅kg⁻¹⋅s⁻²")
    print(f"   Enhanced G:      {G_theory:.6e} m³⋅kg⁻¹⋅s⁻²")
    print(f"   Relative error:  {validation['relative_difference']:.2%}")
    print(f"   Within σ range:  {'✅ Yes' if validation['within_1_sigma'] else '❌ No'}")
    
    # Calculate accuracy
    accuracy = (1 - validation['relative_difference']) * 100
    print(f"   Accuracy:        {accuracy:.1f}%")
    
    # Compare with basic LQG result
    G_basic = 2.987587e-28  # From previous basic calculation
    basic_error = abs(G_basic - G_exp) / G_exp
    enhanced_error = validation['relative_difference']
    
    if enhanced_error > 0:
        improvement = basic_error / enhanced_error
        print(f"   Improvement:     {improvement:.1f}x better than basic LQG")
    
    print(f"\n🎯 Final Enhanced Result:")
    print(f"   G = {G_theory:.6e} m³⋅kg⁻¹⋅s⁻² ({accuracy:.1f}% accurate)")
    
    return results

if __name__ == "__main__":
    main()
