"""
LQG First-Principles Gravitational Constant Derivation Package

This package provides a complete implementation of Newton's gravitational 
constant derivation from Loop Quantum Gravity (LQG) first principles.

Modules:
    scalar_tensor_lagrangian: Enhanced Lagrangian for G → φ(x) framework
    holonomy_flux_algebra: LQG bracket structures with volume corrections
    stress_energy_tensor: Complete T_μν with all corrections
    einstein_field_equations: Modified Einstein equations φ(x)G_μν = 8πT_μν
    gravitational_constant: Final G derivation from LQG parameters

Mathematical Framework:
    G_theoretical = γħc/(8π) × [LQG corrections]
    
    Where γ = 0.2375 is the Barbero-Immirzi parameter.

Author: LQG Research Team
Date: July 2025
"""

from .scalar_tensor_lagrangian import ScalarTensorLagrangian, LQGScalarTensorConfig
from .holonomy_flux_algebra import HolonomyFluxAlgebra, LQGHolonomyFluxConfig
from .stress_energy_tensor import CompleteStressEnergyTensor, StressEnergyConfig
from .einstein_field_equations import EinsteinFieldEquations, EinsteinEquationConfig
from .gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig

# Package metadata
__version__ = "1.0.0"
__author__ = "LQG Research Team"
__email__ = "lqg-research@example.com"
__description__ = "First-principles derivation of Newton's gravitational constant from LQG"

# Key physical constants
BARBERO_IMMIRZI_PARAMETER = 0.2375
PLANCK_LENGTH = 1.616e-35  # meters
PLANCK_TIME = 5.391e-44    # seconds
PLANCK_MASS = 2.176e-8     # kg
EXPERIMENTAL_G = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²

# Main exports
__all__ = [
    # Core classes
    'ScalarTensorLagrangian',
    'HolonomyFluxAlgebra', 
    'CompleteStressEnergyTensor',
    'EinsteinFieldEquations',
    'GravitationalConstantCalculator',
    
    # Configuration classes
    'LQGScalarTensorConfig',
    'LQGHolonomyFluxConfig',
    'StressEnergyConfig', 
    'EinsteinEquationConfig',
    'GravitationalConstantConfig',
    
    # Physical constants
    'BARBERO_IMMIRZI_PARAMETER',
    'PLANCK_LENGTH',
    'PLANCK_TIME', 
    'PLANCK_MASS',
    'EXPERIMENTAL_G'
]

# Convenience function for quick calculations
def quick_G_calculation(gamma=BARBERO_IMMIRZI_PARAMETER, 
                       include_all_corrections=True,
                       verbose=False):
    """
    Quick calculation of gravitational constant from LQG.
    
    Args:
        gamma: Barbero-Immirzi parameter (default: 0.2375)
        include_all_corrections: Include all LQG corrections
        verbose: Print detailed output
        
    Returns:
        Theoretical gravitational constant in SI units
    """
    config = GravitationalConstantConfig(
        gamma_immirzi=gamma,
        include_polymer_corrections=include_all_corrections,
        include_volume_corrections=include_all_corrections,
        include_holonomy_corrections=include_all_corrections,
        verbose_output=verbose
    )
    
    calculator = GravitationalConstantCalculator(config)
    results = calculator.compute_theoretical_G()
    
    if verbose:
        validation = calculator.validate_against_experiment(results)
        print(f"Theoretical G: {results['G_theoretical']:.6e}")
        print(f"Experimental G: {validation['G_experimental']:.6e}")
        print(f"Relative error: {validation['relative_difference_percent']:.2f}%")
    
    return results['G_theoretical']

# Package version check
def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = ['numpy', 'sympy']
    optional_modules = ['matplotlib', 'json']
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_required:
        raise ImportError(f"Missing required dependencies: {missing_required}")
    
    if missing_optional:
        print(f"Warning: Missing optional dependencies: {missing_optional}")
        print("Some features may not be available.")
    
    return True

# Initialize package
def initialize_lqg_package():
    """Initialize the LQG package with dependency checks."""
    try:
        check_dependencies()
        print("LQG First-Principles Gravitational Constant Package initialized successfully!")
        print(f"Version: {__version__}")
        print(f"Barbero-Immirzi parameter: γ = {BARBERO_IMMIRZI_PARAMETER}")
        return True
    except Exception as e:
        print(f"Package initialization failed: {e}")
        return False
