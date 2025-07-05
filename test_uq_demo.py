#!/usr/bin/env python3
"""
Test script to demonstrate the UQ functionality
"""

import sys
import os
sys.path.append('src')

from gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig

def main():
    print("="*60)
    print("LQG GRAVITATIONAL CONSTANT WITH UQ ANALYSIS")
    print("="*60)
    
    # Setup configuration with UQ enabled
    config = GravitationalConstantConfig()
    config.enable_uncertainty_analysis = True
    config.monte_carlo_samples = 1000  # Moderate sample size
    config.confidence_level = 0.95
    
    print(f"Configuration:")
    print(f"  Barbero-Immirzi parameter: {config.gamma_immirzi}")
    print(f"  UQ Analysis: {config.enable_uncertainty_analysis}")
    print(f"  Monte Carlo samples: {config.monte_carlo_samples}")
    print(f"  Confidence level: {config.confidence_level*100}%")
    print()
    
    # Initialize calculator
    print("Initializing LQG gravitational constant calculator...")
    calc = GravitationalConstantCalculator(config)
    print()
    
    # Compute theoretical G
    print("Computing theoretical gravitational constant...")
    theoretical_results = calc.compute_theoretical_G()
    G_theoretical = theoretical_results['G_theoretical_ultra']
    print(f"Theoretical G (without UQ): {G_theoretical:.10e} m³⋅kg⁻¹⋅s⁻²")
    print()
    
    # Perform UQ analysis
    print("CRITICAL UQ ANALYSIS:")
    print("-" * 40)
    uq_results = calc.compute_uncertainty_quantification()
    
    if uq_results['uq_enabled']:
        print(f"UQ Status: {uq_results['uq_status']}")
        print(f"Mean G (with uncertainty): {uq_results['mean_G']:.10e} m³⋅kg⁻¹⋅s⁻²")
        print(f"Standard deviation: ±{uq_results['std_G']:.3e}")
        print(f"Relative uncertainty: {uq_results['relative_uncertainty_percent']:.2f}%")
        print(f"95% confidence interval:")
        print(f"  [{uq_results['confidence_interval_lower']:.10e}, {uq_results['confidence_interval_upper']:.10e}]")
        
        # Check if experimental value is within CI
        experimental_G = 6.67430e-11
        within_ci = uq_results['experimental_within_ci']
        print(f"Experimental G within CI: {'YES' if within_ci else 'NO'}")
        
        # Display top parameter sensitivities
        if 'parameter_sensitivities' in uq_results:
            sensitivities = uq_results['parameter_sensitivities']
            if sensitivities:
                print("\nTop parameter sensitivities:")
                sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
                for param, sens in sorted_sens[:5]:  # Top 5
                    print(f"  {param}: {sens:.3e}")
        
        # UQ quality assessment
        if 'uq_quality_assessment' in uq_results:
            quality = uq_results['uq_quality_assessment']
            print(f"\nUQ Quality: {quality.get('relative_uncertainty_assessment', 'Unknown')}")
            
            # Show recommendations if any critical issues
            if 'recommendations' in quality:
                critical_recs = [r for r in quality['recommendations'] if 'CRITICAL' in r or 'HIGH' in r]
                if critical_recs:
                    print("Critical UQ Recommendations:")
                    for rec in critical_recs:
                        severity = "CRITICAL" if "CRITICAL" in rec else "HIGH"
                        print(f"  [{severity}] {rec}")
                else:
                    print("UQ analysis meets quality standards")
    else:
        print(f"UQ ANALYSIS FAILED")
        if 'error' in uq_results:
            print(f"Error: {uq_results['error']}")
    
    print()
    print("="*60)
    print("UQ ANALYSIS COMPLETE")
    print("="*60)
    
    return calc, uq_results

if __name__ == "__main__":
    calc, uq_results = main()
