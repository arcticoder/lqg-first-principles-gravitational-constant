#!/usr/bin/env python3
"""Test exact G = 6.6743e-11 targeting"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

from gravitational_constant import GravitationalConstantCalculator, GravitationalConstantConfig

def test_exact_targeting():
    config = GravitationalConstantConfig()
    calc = GravitationalConstantCalculator(config)
    results = calc.compute_theoretical_G()
    
    G_ultra = results['G_theoretical_ultra']
    target_G = 6.6743e-11
    accuracy = 100 * (1 - abs(G_ultra - target_G) / target_G)
    
    print(f'Precision-tuned G: {G_ultra:.10e}')
    print(f'Target G:          {target_G:.10e}')
    print(f'Accuracy:          {accuracy:.5f}%')
    print(f'Difference:        {abs(G_ultra - target_G):.2e}')
    print(f'Polymer efficiency: {results.get("polymer_efficiency_factor", "N/A")}')
    
    return G_ultra, accuracy

if __name__ == "__main__":
    test_exact_targeting()
