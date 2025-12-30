import numpy as np
from typing import Tuple, Dict, List
import random

class ParticleDecayChain:
    """
    Models the decay chains of fragile nuclear particles in high-energy collision environments.
    """
    
    def __init__(self):
        # Decay constants for different particle types (in inverse seconds)
        self.decay_constants = {
            'deuteron': 1.0e-21,      # ~10^-21 s
            'antideuteron': 1.0e-21,  # ~10^-21 s
            'neutron': 8.8e6,         # ~8.8 million seconds (8.8e6 s)
            'proton': float('inf'),   # Stable
            'pion': 2.6e16,           # ~2.6e16 s
        }
        
        # Branching ratios for deuteron decay
        self.deuteron_decay_branches = {
            'neutron + proton': 0.999,
            'pion + neutron': 0.001,
        }
        
        # Branching ratios for antideuteron decay
        self.antideuteron_decay_branches = {
            'antineutron + antiproton': 0.999,
            'pion + antineutron': 0.001,
        }
        
        # Energy thresholds for particle formation (GeV)
        self.energy_thresholds = {
            'deuteron': 1.875,       # GeV
            'antideuteron': 1.875,   # GeV
            'neutron': 0.939,        # GeV
            'proton': 0.938,         # GeV
            'pion': 0.135,           # GeV
        }
    
    def calculate_decay_probability(self, particle_type: str, time: float) -> float:
        """
        Calculate probability of decay for a given particle type over time.
        
        Args:
            particle_type: Type of particle ('deuteron', 'antideuteron', etc.)
            time: Time elapsed since creation (seconds)
            
        Returns:
            Probability of decay occurring
        """
        if particle_type not in self.decay_constants:
            return 0.0
            
        decay_constant = self.decay_constants[particle_type]
        
        # Exponential decay probability
        if np.isinf(decay_constant):
            return 0.0  # Stable particle doesn't decay
            
        probability = 1 - np.exp(-decay_constant * time)
        return min(probability, 1.0)  # Clamp between 0 and 1
    
    def simulate_decay_chain(self, initial_particles: Dict[str, int], 
                           time_step: float, total_time: float) -> Dict[str, List[int]]:
        """
        Simulate decay chain for specified initial particles over time.
        
        Args:
            initial_particles: Dictionary of initial particle counts
            time_step: Time step for simulation (seconds)
            total_time: Total simulation time (seconds)
            
        Returns:
            Dictionary tracking particle counts over time
        """
        # Initialize result tracking
        results = {}
        for particle in initial_particles.keys():
            results[particle] = [initial_particles[particle]]
        
        # Track all created particles
        all_particles = initial_particles.copy()
        
        # Simulation loop
        current_time = 0.0
        while current_time < total_time:
            # Create new particles from decay
            new_particles = {}
            
            # Process each particle type
            for particle_type, count in list(all_particles.items()):
                if count <= 0:
                    continue
                    
                # Calculate decay probability
                decay_prob = self.calculate_decay_probability(particle_type, time_step)
                
                # Determine number of decays
                num_decays = np.random.binomial(count, decay_prob)
                
                # Remove decaying particles
                all_particles[particle_type] -= num_decays
                
                # Add decay products
                if particle_type == 'deuteron':
                    decay_products = self._get_deuteron_decay_products(num_decays)
                elif particle_type == 'antideuteron':
                    decay_products = self._get_antideuteron_decay_products(num_decays)
                else:
                    # Other particles don't typically decay into multiple particles
                    decay_products = {particle_type: num_decays}
                
                # Update particle counts
                for product, quantity in decay_products.items():
                    if product not in all_particles:
                        all_particles[product] = 0
                    all_particles[product] += quantity
                    
                    # Track in results
                    if product not in results:
                        results[product] = [0] * len(results[list(results.keys())[0]])
                    results[product].append(all_particles[product])
            
            current_time += time_step
            
            # Update all particle counts in results
            for particle_type in results:
                if particle_type not in all_particles:
                    all_particles[particle_type] = 0
                results[particle_type].append(all_particles[particle_type])
        
        return results
    
    def _get_deuteron_decay_products(self, num_decays: int) -> Dict[str, int]:
        """Get decay products for deuterons."""
        products = {'neutron': 0, 'proton': 0, 'pion': 0, 'antineutron': 0, 'antiproton': 0}
        
        for _ in range(num_decays):
            # Choose decay mode based on branching ratios
            rand_val = random.random()
            cumulative = 0.0
            
            for mode, ratio in self.deuteron_decay_branches.items():
                cumulative += ratio
                if rand_val <= cumulative:
                    if mode == 'neutron + proton':
                        products['neutron'] += 1
                        products['proton'] += 1
                    elif mode == 'pion + neutron':
                        products['pion'] += 1
                        products['neutron'] += 1
                    break
        
        return products
    
    def _get_antideuteron_decay_products(self, num_decays: int) -> Dict[str, int]:
        """Get decay products for antideuterons."""
        products = {'neutron': 0, 'proton': 0, 'pion': 0, 'antineutron': 0, 'antiproton': 0}
        
        for _ in range(num_decays):
            # Choose decay mode based on branching ratios
            rand_val = random.random()
            cumulative = 0.0
            
            for mode, ratio in self.antideuteron_decay_branches.items():
                cumulative += ratio
                if rand_val <= cumulative:
                    if mode == 'antineutron + antiproton':
                        products['antineutron'] += 1
                        products['antiproton'] += 1
                    elif mode == 'pion + antineutron':
                        products['pion'] += 1
                        products['antineutron'] += 1
                    break
        
        return products
    
    def is_formation_possible(self, energy: float, particle_type: str) -> bool:
        """
        Check if formation of a particle is possible given available energy.
        
        Args:
            energy: Available energy (GeV)
            particle_type: Type of particle to form
            
        Returns:
            True if formation is possible, False otherwise
        """
        threshold = self.energy_thresholds.get(particle_type, 0.0)
        return energy >= threshold
    
    def get_production_rate(self, temperature: float, particle_type: str) -> float:
        """
        Estimate production rate based on temperature (simplified model).
        
        Args:
            temperature: System temperature (Kelvin)
            particle_type: Type of particle
            
        Returns:
            Production rate (arbitrary units)
        """
        # Simplified exponential dependence on temperature
        base_energy = self.energy_thresholds.get(particle_type, 1.0)
        # Convert temperature to GeV (approximate conversion)
        temp_gev = temperature / 1.16e10  # Kelvin to GeV
        
        if temp_gev < base_energy:
            return 0.0
            
        # Simple exponential dependence
        rate = np.exp((temp_gev - base_energy) / base_energy) * 1e-30
        return max(rate, 0.0)