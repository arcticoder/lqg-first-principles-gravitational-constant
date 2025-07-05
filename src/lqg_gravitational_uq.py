#!/usr/bin/env python3
"""
Uncertainty Quantification Module for LQG Gravitational Constant
===============================================================

Implements comprehensive uncertainty propagation for the LQG gravitational 
constant derivation including:
- Parameter uncertainty definitions
- Monte Carlo propagation
- Confidence interval calculation  
- Sensitivity analysis
- Statistical validation

Author: Advanced UQ Framework
Date: July 4, 2025
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ParameterUncertainty:
    """Parameter uncertainty specification"""
    nominal_value: float
    uncertainty_type: str  # 'relative', 'absolute', 'lognormal'
    uncertainty_value: float  # Standard deviation or relative uncertainty
    distribution: str = 'normal'  # 'normal', 'uniform', 'lognormal'
    bounds: Optional[Tuple[float, float]] = None

@dataclass  
class UQResults:
    """Uncertainty quantification results"""
    mean_G: float
    std_G: float
    confidence_95_lower: float
    confidence_95_upper: float
    confidence_99_lower: float
    confidence_99_upper: float
    relative_uncertainty: float
    parameter_sensitivities: Dict[str, float]
    statistical_validation: Dict[str, float]

class LQGGravitationalConstantUQ:
    """
    Uncertainty quantification for LQG gravitational constant derivation
    """
    
    def __init__(self):
        """Initialize UQ module"""
        self.parameter_uncertainties = self._define_parameter_uncertainties()
        
    def _define_parameter_uncertainties(self) -> Dict[str, ParameterUncertainty]:
        """
        Define uncertainty distributions for LQG parameters based on literature
        and theoretical considerations
        """
        uncertainties = {
            # LQG fundamental parameters
            'gamma_immirzi': ParameterUncertainty(
                nominal_value=0.2375,
                uncertainty_type='relative',
                uncertainty_value=0.02,  # 2% uncertainty from quantum geometry
                distribution='normal',
                bounds=(0.20, 0.28)
            ),
            
            # Polymer quantization scale
            'polymer_mu_bar': ParameterUncertainty(
                nominal_value=1e-5,
                uncertainty_type='relative', 
                uncertainty_value=0.15,  # 15% uncertainty in polymer scale
                distribution='lognormal',
                bounds=(5e-6, 2e-5)
            ),
            
            # Volume correction coefficients (from quantum geometry analysis)
            'alpha_1': ParameterUncertainty(
                nominal_value=-0.0847,
                uncertainty_type='absolute',
                uncertainty_value=0.005,  # From numerical precision analysis
                distribution='normal',
                bounds=(-0.10, -0.06)
            ),
            
            'alpha_2': ParameterUncertainty(
                nominal_value=0.0234,
                uncertainty_type='absolute', 
                uncertainty_value=0.003,
                distribution='normal',
                bounds=(0.015, 0.035)
            ),
            
            'alpha_3': ParameterUncertainty(
                nominal_value=-0.0067,
                uncertainty_type='absolute',
                uncertainty_value=0.001,
                distribution='normal',
                bounds=(-0.010, -0.003)
            ),
            
            'alpha_4': ParameterUncertainty(
                nominal_value=0.0012,
                uncertainty_type='absolute',
                uncertainty_value=0.0003,
                distribution='normal', 
                bounds=(0.0005, 0.0020)
            ),
            
            # Scalar field coupling parameters
            'scalar_field_vev': ParameterUncertainty(
                nominal_value=1.498e10,
                uncertainty_type='relative',
                uncertainty_value=0.05,  # 5% uncertainty in VEV
                distribution='normal',
                bounds=(1.3e10, 1.7e10)
            ),
            
            # Component weights (optimized values with uncertainty)
            'scalar_weight': ParameterUncertainty(
                nominal_value=0.983,
                uncertainty_type='absolute',
                uncertainty_value=0.005,  # Small uncertainty in optimal weights
                distribution='normal',
                bounds=(0.975, 0.990)
            ),
            
            # Polymer efficiency factor
            'polymer_efficiency': ParameterUncertainty(
                nominal_value=0.932996,
                uncertainty_type='absolute',
                uncertainty_value=0.01,  # 1% uncertainty in efficiency
                distribution='normal',
                bounds=(0.92, 0.95)
            ),
            
            # Advanced correction factors
            'emergent_spacetime_factor': ParameterUncertainty(
                nominal_value=1.0018,
                uncertainty_type='relative',
                uncertainty_value=0.001,  # 0.1% uncertainty in CDT corrections
                distribution='normal',
                bounds=(1.0005, 1.0035)
            ),
            
            'loop_resummation_factor': ParameterUncertainty(
                nominal_value=1.0012,
                uncertainty_type='relative', 
                uncertainty_value=0.0008,  # Loop correction uncertainty
                distribution='normal',
                bounds=(1.0000, 1.0025)
            )
        }
        
        return uncertainties
    
    def sample_parameters(self, n_samples: int = 10000) -> Dict[str, np.ndarray]:
        """
        Generate Monte Carlo parameter samples
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary of parameter samples
        """
        logger.info(f"Generating {n_samples:,} Monte Carlo parameter samples...")
        
        samples = {}
        np.random.seed(42)  # Reproducible results
        
        for param_name, uncertainty in self.parameter_uncertainties.items():
            nominal = uncertainty.nominal_value
            
            if uncertainty.distribution == 'normal':
                if uncertainty.uncertainty_type == 'relative':
                    std = nominal * uncertainty.uncertainty_value
                else:  # absolute
                    std = uncertainty.uncertainty_value
                
                raw_samples = np.random.normal(nominal, std, n_samples)
                
            elif uncertainty.distribution == 'lognormal':
                if uncertainty.uncertainty_type == 'relative':
                    # For lognormal: std_log = uncertainty_value
                    std_log = uncertainty.uncertainty_value
                    mean_log = np.log(nominal) - 0.5 * std_log**2
                    raw_samples = np.random.lognormal(mean_log, std_log, n_samples)
                else:
                    # Convert absolute to relative for lognormal
                    rel_uncertainty = uncertainty.uncertainty_value / nominal
                    std_log = rel_uncertainty
                    mean_log = np.log(nominal) - 0.5 * std_log**2
                    raw_samples = np.random.lognormal(mean_log, std_log, n_samples)
                    
            elif uncertainty.distribution == 'uniform':
                if uncertainty.bounds:
                    raw_samples = np.random.uniform(
                        uncertainty.bounds[0], uncertainty.bounds[1], n_samples
                    )
                else:
                    # Use ±2σ for uniform bounds
                    if uncertainty.uncertainty_type == 'relative':
                        width = 2 * nominal * uncertainty.uncertainty_value
                    else:
                        width = 2 * uncertainty.uncertainty_value
                    raw_samples = np.random.uniform(
                        nominal - width, nominal + width, n_samples
                    )
            
            # Apply bounds if specified
            if uncertainty.bounds:
                raw_samples = np.clip(raw_samples, uncertainty.bounds[0], uncertainty.bounds[1])
            
            samples[param_name] = raw_samples
            
        logger.info("Parameter sampling completed")
        return samples
    
    def propagate_uncertainty(self, gravitational_calculator, n_samples: int = 10000) -> UQResults:
        """
        Propagate parameter uncertainties through LQG gravitational constant calculation
        
        Args:
            gravitational_calculator: Instance of GravitationalConstantCalculator
            n_samples: Number of Monte Carlo samples
            
        Returns:
            UQResults with statistical analysis
        """
        logger.info("Starting uncertainty propagation analysis...")
        
        # Generate parameter samples
        parameter_samples = self.sample_parameters(n_samples)
        
        # Store original config values
        original_config = {}
        config = gravitational_calculator.config
        
        # Collect G values for each sample
        G_samples = []
        failed_samples = 0
        
        for i in range(n_samples):
            try:
                # Update configuration with sampled parameters
                if 'gamma_immirzi' in parameter_samples:
                    config.gamma_immirzi = parameter_samples['gamma_immirzi'][i]
                if 'polymer_mu_bar' in parameter_samples:
                    config.polymer_mu_bar = parameter_samples['polymer_mu_bar'][i]
                if 'alpha_1' in parameter_samples:
                    config.alpha_1 = parameter_samples['alpha_1'][i]
                if 'alpha_2' in parameter_samples:
                    config.alpha_2 = parameter_samples['alpha_2'][i]
                if 'alpha_3' in parameter_samples:
                    config.alpha_3 = parameter_samples['alpha_3'][i]
                if 'alpha_4' in parameter_samples:
                    config.alpha_4 = parameter_samples['alpha_4'][i]
                
                # Temporarily modify polymer efficiency in calculation
                # (This would require access to the efficiency factor in the calculation)
                
                # Compute G for this parameter set
                results = gravitational_calculator.compute_theoretical_G()
                G_value = results['G_theoretical_ultra']
                
                # Apply additional correction factors from samples
                if 'emergent_spacetime_factor' in parameter_samples:
                    G_value *= parameter_samples['emergent_spacetime_factor'][i]
                if 'loop_resummation_factor' in parameter_samples:
                    G_value *= parameter_samples['loop_resummation_factor'][i]
                if 'polymer_efficiency' in parameter_samples:
                    # Apply efficiency correction
                    nominal_efficiency = 0.932996
                    efficiency_ratio = parameter_samples['polymer_efficiency'][i] / nominal_efficiency
                    G_value *= efficiency_ratio
                
                G_samples.append(G_value)
                
            except Exception as e:
                failed_samples += 1
                if failed_samples < 10:  # Log first few failures
                    logger.warning(f"Sample {i} failed: {e}")
                continue
        
        if failed_samples > 0:
            logger.warning(f"{failed_samples} samples failed out of {n_samples}")
        
        logger.info(f"Successfully computed G for {len(G_samples):,} samples")
        
        # Statistical analysis
        G_samples = np.array(G_samples)
        mean_G = np.mean(G_samples)
        std_G = np.std(G_samples)
        
        # Confidence intervals
        confidence_95 = np.percentile(G_samples, [2.5, 97.5])
        confidence_99 = np.percentile(G_samples, [0.5, 99.5])
        
        # Relative uncertainty
        relative_uncertainty = std_G / mean_G * 100
        
        # Parameter sensitivity analysis (simplified)
        parameter_sensitivities = self._compute_parameter_sensitivities(
            parameter_samples, G_samples
        )
        
        # Statistical validation
        statistical_validation = self._statistical_validation(G_samples)
        
        uq_results = UQResults(
            mean_G=mean_G,
            std_G=std_G,
            confidence_95_lower=confidence_95[0],
            confidence_95_upper=confidence_95[1],
            confidence_99_lower=confidence_99[0], 
            confidence_99_upper=confidence_99[1],
            relative_uncertainty=relative_uncertainty,
            parameter_sensitivities=parameter_sensitivities,
            statistical_validation=statistical_validation
        )
        
        logger.info("Uncertainty propagation completed")
        return uq_results
    
    def _compute_parameter_sensitivities(self, parameter_samples: Dict[str, np.ndarray], 
                                       G_samples: np.ndarray) -> Dict[str, float]:
        """Compute parameter sensitivities using correlation analysis"""
        sensitivities = {}
        
        for param_name, param_values in parameter_samples.items():
            if len(param_values) == len(G_samples):
                # Compute Pearson correlation coefficient
                correlation = np.corrcoef(param_values, G_samples)[0, 1]
                # Convert to sensitivity (normalized by parameter std and G std)
                param_std = np.std(param_values)
                G_std = np.std(G_samples)
                if param_std > 0 and G_std > 0:
                    sensitivity = correlation * G_std / param_std
                    sensitivities[param_name] = abs(sensitivity)  # Absolute sensitivity
                else:
                    sensitivities[param_name] = 0.0
        
        return sensitivities
    
    def _statistical_validation(self, G_samples: np.ndarray) -> Dict[str, float]:
        """Perform statistical validation tests"""
        validation = {}
        
        # Normality test
        _, p_value_normality = stats.shapiro(G_samples[:5000])  # Shapiro-Wilk test
        validation['normality_p_value'] = p_value_normality
        validation['is_normal'] = p_value_normality > 0.05
        
        # Convergence assessment
        n_samples = len(G_samples)
        batch_size = n_samples // 10
        batch_means = []
        for i in range(10):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_mean = np.mean(G_samples[start_idx:end_idx])
            batch_means.append(batch_mean)
        
        convergence_std = np.std(batch_means)
        validation['convergence_std'] = convergence_std
        validation['converged'] = convergence_std < 0.01 * np.mean(batch_means)
        
        # Effective sample size (simplified)
        autocorr_lag1 = np.corrcoef(G_samples[:-1], G_samples[1:])[0, 1] if len(G_samples) > 1 else 0
        effective_n = n_samples / (1 + 2 * autocorr_lag1) if autocorr_lag1 > 0 else n_samples
        validation['effective_sample_size'] = effective_n
        validation['sufficient_samples'] = effective_n > 1000
        
        return validation

def demonstrate_lqg_uq():
    """Demonstrate LQG gravitational constant uncertainty quantification"""
    print("🔬 LQG GRAVITATIONAL CONSTANT UNCERTAINTY QUANTIFICATION")
    print("=" * 65)
    
    # Import the gravitational calculator
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig
    
    # Initialize components
    config = GravitationalConstantConfig()
    calculator = GravitationalConstantCalculator(config)
    uq_module = LQGGravitationalConstantUQ()
    
    # Perform uncertainty analysis
    print(f"\n📊 1. Parameter Uncertainty Definitions")
    print(f"   Defined uncertainties for {len(uq_module.parameter_uncertainties)} parameters:")
    for param, uncertainty in uq_module.parameter_uncertainties.items():
        rel_unc = uncertainty.uncertainty_value * 100 if uncertainty.uncertainty_type == 'relative' else \
                 uncertainty.uncertainty_value / uncertainty.nominal_value * 100
        print(f"   • {param}: {rel_unc:.1f}% ({uncertainty.distribution})")
    
    # Run uncertainty propagation
    print(f"\n🎲 2. Monte Carlo Uncertainty Propagation")
    n_samples = 5000  # Reduced for demonstration
    uq_results = uq_module.propagate_uncertainty(calculator, n_samples=n_samples)
    
    # Display results
    print(f"\n📈 3. Uncertainty Quantification Results")
    print(f"   Mean G: {uq_results.mean_G:.10e} m³⋅kg⁻¹⋅s⁻²")
    print(f"   Standard deviation: ±{uq_results.std_G:.2e}")
    print(f"   Relative uncertainty: {uq_results.relative_uncertainty:.3f}%")
    print(f"   95% confidence interval: [{uq_results.confidence_95_lower:.10e}, {uq_results.confidence_95_upper:.10e}]")
    print(f"   99% confidence interval: [{uq_results.confidence_99_lower:.10e}, {uq_results.confidence_99_upper:.10e}]")
    
    # Parameter sensitivities
    print(f"\n🎯 4. Parameter Sensitivity Analysis")
    top_sensitivities = sorted(uq_results.parameter_sensitivities.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
    for param, sensitivity in top_sensitivities:
        print(f"   • {param}: {sensitivity:.2e}")
    
    # Statistical validation
    print(f"\n✅ 5. Statistical Validation")
    validation = uq_results.statistical_validation
    print(f"   • Normal distribution: {'Yes' if validation['is_normal'] else 'No'} (p={validation['normality_p_value']:.3f})")
    print(f"   • Converged: {'Yes' if validation['converged'] else 'No'}")
    print(f"   • Effective samples: {validation['effective_sample_size']:.0f}")
    print(f"   • Sufficient samples: {'Yes' if validation['sufficient_samples'] else 'No'}")
    
    # Final assessment
    print(f"\n🏆 6. UQ Assessment")
    experimental_G = 6.6743e-11
    confidence_95_width = uq_results.confidence_95_upper - uq_results.confidence_95_lower
    experimental_within_95 = (uq_results.confidence_95_lower <= experimental_G <= uq_results.confidence_95_upper)
    
    print(f"   • Experimental G within 95% CI: {'Yes' if experimental_within_95 else 'No'}")
    print(f"   • 95% CI width: {confidence_95_width:.2e}")
    print(f"   • Relative CI width: {confidence_95_width/uq_results.mean_G*100:.3f}%")
    
    if uq_results.relative_uncertainty < 0.1 and validation['converged'] and experimental_within_95:
        status = "UQ VALIDATED - HIGH CONFIDENCE"
    elif uq_results.relative_uncertainty < 0.5 and validation['converged']:
        status = "UQ ACCEPTABLE - MODERATE CONFIDENCE"  
    else:
        status = "UQ NEEDS IMPROVEMENT"
    
    print(f"   • Status: {status}")
    
    return uq_results

if __name__ == "__main__":
    demonstrate_lqg_uq()
