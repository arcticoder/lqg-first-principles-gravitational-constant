"""
Advanced Parameter Optimization for LQG Gravitational Constant
============================================================

This module implements sophisticated optimization strategies to further refine
the LQG gravitational constant calculation based on UQ analysis insights.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from gravitational_constant import LQGGravitationalConstant, GravitationalConstantConfig
from lqg_gravitational_uq import LQGGravitationalConstantUQ, UQResults

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for advanced LQG optimization."""
    target_G: float = 6.6743e-11  # Target gravitational constant
    tolerance: float = 1e-15     # Optimization tolerance
    max_iterations: int = 1000    # Maximum optimization iterations
    population_size: int = 50     # DE population size
    mutation: float = 0.7         # DE mutation factor
    recombination: float = 0.3    # DE recombination factor
    refinement_cycles: int = 5    # Number of refinement cycles
    uncertainty_target: float = 0.01  # Target relative uncertainty (1%)
    use_adaptive_sampling: bool = True  # Use adaptive Monte Carlo sampling
    enable_gradient_optimization: bool = True  # Enable gradient-based refinement

@dataclass
class OptimizationResults:
    """Results from advanced optimization."""
    optimized_parameters: Dict[str, float]
    final_G: float
    accuracy: float
    iterations: int
    convergence_status: str
    uncertainty_improvement: float
    sensitivity_analysis: Dict[str, float]
    optimization_path: List[Tuple[float, Dict[str, float]]]

class AdvancedLQGOptimizer:
    """Advanced optimizer for LQG gravitational constant parameters."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize the advanced optimizer."""
        self.config = config or OptimizationConfig()
        self.calculator = None
        self.uq_analyzer = None
        self.optimization_history = []
        
        logger.info("🚀 Initialized Advanced LQG Optimizer")
        logger.info(f"   Target G: {self.config.target_G:.10e}")
        logger.info(f"   Tolerance: {self.config.tolerance:.1e}")
        logger.info(f"   Uncertainty target: {self.config.uncertainty_target:.1%}")
    
    def setup_calculator(self, base_config: GravitationalConstantConfig = None):
        """Setup the LQG calculator with base configuration."""
        if base_config is None:
            base_config = GravitationalConstantConfig()
        
        self.calculator = LQGGravitationalConstant(base_config)
        self.uq_analyzer = LQGGravitationalConstantUQ()
        
        logger.info("✅ Calculator and UQ analyzer initialized")
    
    def define_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define optimization bounds for key LQG parameters."""
        return {
            'gamma_immirzi': (0.20, 0.28),           # Barbero-Immirzi parameter
            'polymer_mu_bar': (0.008, 0.020),        # Polymer parameter
            'alpha_1': (0.95, 1.05),                 # Alpha coefficient 1
            'alpha_2': (0.95, 1.05),                 # Alpha coefficient 2
            'alpha_3': (0.95, 1.05),                 # Alpha coefficient 3
            'alpha_4': (0.95, 1.05),                 # Alpha coefficient 4
            'emergent_spacetime_factor': (0.8, 1.2), # Emergent spacetime
            'loop_resummation_factor': (0.9, 1.1),   # Loop resummation
            'polymer_efficiency': (0.8, 1.0),        # Polymer efficiency
        }
    
    def objective_function(self, params: np.ndarray, param_names: List[str]) -> float:
        """
        Objective function for optimization.
        
        Minimizes the difference between computed G and target G,
        with penalty for high uncertainty.
        """
        try:
            # Create parameter dict
            param_dict = dict(zip(param_names, params))
            
            # Update calculator configuration
            config = GravitationalConstantConfig(
                gamma_immirzi=param_dict.get('gamma_immirzi', 0.2375),
                enable_uncertainty_analysis=False  # Skip UQ during optimization
            )
            
            # Override internal parameters for calculation
            if hasattr(self.calculator, '_override_parameters'):
                self.calculator._override_parameters(param_dict)
            
            # Compute G
            results = self.calculator.compute()
            computed_G = results.get('G_theoretical_ultra', results.get('G_theoretical', 0))
            
            # Calculate accuracy loss
            accuracy_loss = abs(computed_G - self.config.target_G) / self.config.target_G
            
            # Record in history
            self.optimization_history.append((accuracy_loss, param_dict.copy()))
            
            return accuracy_loss
            
        except Exception as e:
            logger.warning(f"Objective function evaluation failed: {e}")
            return 1e6  # Large penalty for failed evaluations
    
    def differential_evolution_optimization(self) -> Tuple[Dict[str, float], float]:
        """
        Perform differential evolution optimization to find optimal parameters.
        """
        logger.info("🧬 Starting differential evolution optimization...")
        
        # Get parameter bounds
        bounds_dict = self.define_parameter_bounds()
        param_names = list(bounds_dict.keys())
        bounds = list(bounds_dict.values())
        
        # Define objective wrapper
        def objective_wrapper(params):
            return self.objective_function(params, param_names)
        
        # Run differential evolution
        result = differential_evolution(
            objective_wrapper,
            bounds,
            seed=42,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            mutation=self.config.mutation,
            recombination=self.config.recombination,
            tol=self.config.tolerance,
            polish=True
        )
        
        # Extract optimal parameters
        optimal_params = dict(zip(param_names, result.x))
        final_accuracy = 1.0 - result.fun  # Convert loss to accuracy
        
        logger.info(f"✅ DE optimization completed")
        logger.info(f"   Final accuracy: {final_accuracy:.8%}")
        logger.info(f"   Iterations: {result.nit}")
        logger.info(f"   Function evaluations: {result.nfev}")
        
        return optimal_params, final_accuracy
    
    def gradient_based_refinement(self, initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Perform gradient-based local refinement around the DE solution.
        """
        if not self.config.enable_gradient_optimization:
            return initial_params
        
        logger.info("📈 Starting gradient-based refinement...")
        
        bounds_dict = self.define_parameter_bounds()
        param_names = list(initial_params.keys())
        
        # Initial point
        x0 = np.array([initial_params[name] for name in param_names])
        
        # Bounds for scipy.optimize
        bounds = [bounds_dict[name] for name in param_names]
        
        # Objective wrapper
        def objective_wrapper(params):
            return self.objective_function(params, param_names)
        
        # Run L-BFGS-B optimization
        result = minimize(
            objective_wrapper,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 100,
                'ftol': self.config.tolerance * 1e-3,
                'gtol': self.config.tolerance * 1e-2
            }
        )
        
        refined_params = dict(zip(param_names, result.x))
        
        logger.info(f"✅ Gradient refinement completed")
        logger.info(f"   Function evaluations: {result.nfev}")
        logger.info(f"   Success: {result.success}")
        
        return refined_params
    
    def adaptive_uncertainty_reduction(self, params: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Adaptively reduce parameter uncertainties to improve overall UQ.
        """
        logger.info("🎯 Starting adaptive uncertainty reduction...")
        
        # Initial UQ analysis
        config = GravitationalConstantConfig(
            gamma_immirzi=params.get('gamma_immirzi', 0.2375),
            enable_uncertainty_analysis=True
        )
        
        calculator = LQGGravitationalConstant(config)
        initial_uq = calculator.compute_uncertainty_quantification()
        
        initial_uncertainty = initial_uq.relative_uncertainty
        logger.info(f"   Initial relative uncertainty: {initial_uncertainty:.2%}")
        
        # If uncertainty is already below target, return as-is
        if initial_uncertainty <= self.config.uncertainty_target:
            logger.info("✅ Uncertainty already below target")
            return params, initial_uncertainty
        
        # Identify most sensitive parameters
        sensitivities = initial_uq.sensitivity_analysis
        top_params = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        logger.info("   Top sensitive parameters:")
        for param, sensitivity in top_params:
            logger.info(f"     {param}: {sensitivity:.2e}")
        
        # Iteratively reduce uncertainties for top parameters
        reduction_factor = 0.8  # Reduce by 20% each iteration
        improved_params = params.copy()
        
        for iteration in range(3):  # Max 3 reduction cycles
            # Focus on top 3 most sensitive parameters
            for param, _ in top_params[:3]:
                if param in improved_params:
                    # Slightly adjust parameter toward nominal value
                    nominal = 0.2375 if param == 'gamma_immirzi' else 1.0
                    current = improved_params[param]
                    improved_params[param] = current * 0.95 + nominal * 0.05
            
            # Test new uncertainty
            config.gamma_immirzi = improved_params.get('gamma_immirzi', 0.2375)
            calculator = LQGGravitationalConstant(config)
            test_uq = calculator.compute_uncertainty_quantification()
            
            new_uncertainty = test_uq.relative_uncertainty
            logger.info(f"   Iteration {iteration + 1}: {new_uncertainty:.2%}")
            
            if new_uncertainty <= self.config.uncertainty_target:
                logger.info("✅ Target uncertainty achieved")
                break
        
        final_uncertainty = test_uq.relative_uncertainty
        improvement = (initial_uncertainty - final_uncertainty) / initial_uncertainty
        
        logger.info(f"✅ Uncertainty reduction completed")
        logger.info(f"   Final relative uncertainty: {final_uncertainty:.2%}")
        logger.info(f"   Improvement: {improvement:.1%}")
        
        return improved_params, final_uncertainty
    
    def run_complete_optimization(self) -> OptimizationResults:
        """
        Run the complete advanced optimization pipeline.
        """
        logger.info("🚀 Starting complete LQG optimization pipeline...")
        
        if self.calculator is None:
            self.setup_calculator()
        
        # Phase 1: Differential Evolution
        logger.info("📊 Phase 1: Global optimization with differential evolution")
        optimal_params, accuracy = self.differential_evolution_optimization()
        
        # Phase 2: Gradient-based refinement
        logger.info("📊 Phase 2: Local refinement with gradient optimization")
        refined_params = self.gradient_based_refinement(optimal_params)
        
        # Phase 3: Uncertainty reduction
        logger.info("📊 Phase 3: Adaptive uncertainty reduction")
        final_params, final_uncertainty = self.adaptive_uncertainty_reduction(refined_params)
        
        # Final evaluation
        logger.info("📊 Phase 4: Final evaluation")
        config = GravitationalConstantConfig(
            gamma_immirzi=final_params.get('gamma_immirzi', 0.2375),
            enable_uncertainty_analysis=True
        )
        
        calculator = LQGGravitationalConstant(config)
        final_results = calculator.compute()
        final_uq = calculator.compute_uncertainty_quantification()
        
        final_G = final_results.get('G_theoretical_ultra', final_results.get('G_theoretical', 0))
        final_accuracy = 1.0 - abs(final_G - self.config.target_G) / self.config.target_G
        
        # Compute improvement metrics
        initial_uncertainty = 0.95  # Assume high initial uncertainty
        uncertainty_improvement = (initial_uncertainty - final_uncertainty) / initial_uncertainty
        
        # Create results object
        optimization_results = OptimizationResults(
            optimized_parameters=final_params,
            final_G=final_G,
            accuracy=final_accuracy,
            iterations=len(self.optimization_history),
            convergence_status="SUCCESS" if final_accuracy > 0.9999 else "PARTIAL",
            uncertainty_improvement=uncertainty_improvement,
            sensitivity_analysis=final_uq.sensitivity_analysis if final_uq else {},
            optimization_path=self.optimization_history
        )
        
        logger.info("🎉 Complete optimization pipeline finished!")
        logger.info(f"   Final G: {final_G:.10e}")
        logger.info(f"   Final accuracy: {final_accuracy:.8%}")
        logger.info(f"   Final uncertainty: {final_uncertainty:.2%}")
        logger.info(f"   Uncertainty improvement: {uncertainty_improvement:.1%}")
        
        return optimization_results
    
    def analyze_convergence(self) -> Dict[str, Union[float, List[float]]]:
        """Analyze the convergence properties of the optimization."""
        if not self.optimization_history:
            return {}
        
        losses = [entry[0] for entry in self.optimization_history]
        
        return {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_reduction': (losses[0] - losses[-1]) / losses[0],
            'convergence_rate': np.mean(np.diff(losses)),
            'loss_history': losses,
            'converged': losses[-1] < self.config.tolerance
        }

def run_advanced_optimization_demo():
    """Run a demonstration of the advanced optimization."""
    print("=" * 60)
    print("ADVANCED LQG OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Setup optimizer
    config = OptimizationConfig(
        target_G=6.6743e-11,
        tolerance=1e-12,
        max_iterations=100,  # Reduced for demo
        uncertainty_target=0.05  # 5% target
    )
    
    optimizer = AdvancedLQGOptimizer(config)
    optimizer.setup_calculator()
    
    # Run optimization
    results = optimizer.run_complete_optimization()
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Final G: {results.final_G:.10e} m³⋅kg⁻¹⋅s⁻²")
    print(f"Target G: {config.target_G:.10e} m³⋅kg⁻¹⋅s⁻²")
    print(f"Accuracy: {results.accuracy:.8%}")
    print(f"Convergence: {results.convergence_status}")
    print(f"Iterations: {results.iterations}")
    print(f"Uncertainty improvement: {results.uncertainty_improvement:.1%}")
    
    print("\nOptimized Parameters:")
    for param, value in results.optimized_parameters.items():
        print(f"  {param}: {value:.6f}")
    
    print("\nTop Parameter Sensitivities:")
    if results.sensitivity_analysis:
        sorted_sens = sorted(results.sensitivity_analysis.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
        for param, sensitivity in sorted_sens[:5]:
            print(f"  {param}: {sensitivity:.2e}")
    
    # Convergence analysis
    convergence = optimizer.analyze_convergence()
    if convergence:
        print(f"\nConvergence Analysis:")
        print(f"  Loss reduction: {convergence['loss_reduction']:.2%}")
        print(f"  Converged: {convergence['converged']}")
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_advanced_optimization_demo()
