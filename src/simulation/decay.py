import numpy as np
from typing import Dict, List, Tuple
import random

class ParticleDecayChain:
    """
    Models the decay chains of fragile nuclear particles in high-energy collision environments.
    """
    
    def __init__(self):
        # Decay constants for different particle species (in inverse seconds)
        self.decay_constants = {
            'deuteron': 1.0e-21,      # ~10^-21 s
            'antideuteron': 1.0e-21,  # ~10^-21 s
            'neutron': 8.8e6,         # ~8.8 million seconds (about 102 days)
            'proton': float('inf'),   # Stable
            'pion': 2.6e16,           # ~2.6e16 s (2.6e16 seconds)
        }
        
        # Decay branching ratios (sum to 1.0)
        self.decay_branching_ratios = {
            'deuteron': {'neutron+proton': 1.0},
            'antideuteron': {'antineutron+antiproton': 1.0},
            'neutron': {'proton+electron+antineutrino': 0.75,
                       'proton+positron+neutrino': 0.25},
            'pion': {'muon+neutrino': 0.99},
        }
        
        # Energy thresholds for particle formation (GeV)
        self.energy_thresholds = {
            'deuteron': 1.875,      # GeV
            'antideuteron': 1.875,  # GeV
            'neutron': 0.939,       # GeV
            'proton': 0.938,        # GeV
            'pion': 0.135,          # GeV
        }
    
    def simulate_decay(self, particle_type: str, time_step: float, 
                      initial_count: int) -> Tuple[int, Dict[str, int]]:
        """
        Simulate decay of particles over a time step.
        
        Args:
            particle_type: Type of particle to decay ('deuteron', 'antideuteron', etc.)
            time_step: Time step in seconds
            initial_count: Initial number of particles
            
        Returns:
            Tuple of (remaining_count, decay_products)
        """
        if particle_type not in self.decay_constants:
            raise ValueError(f"Unknown particle type: {particle_type}")
            
        decay_constant = self.decay_constants[particle_type]
        
        # Calculate probability of decay using exponential decay formula
        if decay_constant == float('inf'):
            # Stable particle - no decay
            return initial_count, {}
        
        # Probability of decay in this time step
        decay_probability = 1.0 - np.exp(-decay_constant * time_step)
        
        # Number of particles that decay
        decayed_count = np.random.binomial(initial_count, decay_probability)
        
        # Remaining particles
        remaining_count = initial_count - decayed_count
        
        # Determine decay products based on branching ratios
        decay_products = self._determine_decay_products(particle_type, decayed_count)
        
        return remaining_count, decay_products
    
    def _determine_decay_products(self, particle_type: str, decayed_count: int) -> Dict[str, int]:
        """
        Determine the decay products based on branching ratios.
        """
        if particle_type not in self.decay_branching_ratios:
            return {}
            
        branching_ratios = self.decay_branching_ratios[particle_type]
        
        # Choose decay mode based on branching ratios
        decay_modes = list(branching_ratios.keys())
        probabilities = list(branching_ratios.values())
        
        # Select decay mode for all decaying particles
        selected_mode = np.random.choice(decay_modes, p=probabilities)
        
        # For simplicity, we assume each decay produces one product particle
        # In reality, this would be more complex
        if selected_mode == 'neutron+proton':
            return {'neutron': decayed_count, 'proton': decayed_count}
        elif selected_mode == 'antineutron+antiproton':
            return {'antineutron': decayed_count, 'antiproton': decayed_count}
        elif selected_mode == 'proton+electron+antineutrino':
            return {'proton': decayed_count, 'electron': decayed_count, 'antineutrino': decayed_count}
        elif selected_mode == 'proton+positron+neutrino':
            return {'proton': decayed_count, 'positron': decayed_count, 'neutrino': decayed_count}
        elif selected_mode == 'muon+neutrino':
            return {'muon': decayed_count, 'neutrino': decayed_count}
        else:
            return {}
    
    def calculate_formation_probability(self, temperature: float, particle_type: str) -> float:
        """
        Calculate the probability of forming a particle given the current temperature.
        
        Args:
            temperature: Current temperature in Kelvin
            particle_type: Type of particle to form
            
        Returns:
            Formation probability (0.0 to 1.0)
        """
        # Convert temperature to GeV for consistency with energy thresholds
        # 1 K ≈ 8.617e-14 GeV
        temp_gev = temperature * 8.617e-14
        
        threshold = self.energy_thresholds.get(particle_type, 0.0)
        
        if threshold <= 0:
            return 0.0
            
        # Simple exponential dependence on temperature relative to threshold
        # Higher temperatures increase formation probability
        if temp_gev > threshold:
            # Probability increases exponentially with excess energy
            excess_energy = temp_gev / threshold
            probability = 1.0 - np.exp(-excess_energy)
            return min(probability, 1.0)
        else:
            # Below threshold, very low probability
            return 0.0
    
    def get_particle_mass(self, particle_type: str) -> float:
        """
        Get the mass of a particle in GeV/c^2.
        
        Args:
            particle_type: Type of particle
            
        Returns:
            Mass in GeV/c^2
        """
        masses = {
            'deuteron': 1.875,
            'antideuteron': 1.875,
            'neutron': 0.939,
            'proton': 0.938,
            'pion': 0.135,
            'electron': 0.000511,
            'positron': 0.000511,
            'muon': 0.106,
        }
        return masses.get(particle_type, 0.0)
    
    def simulate_collision_event(self, initial_particles: Dict[str, int], 
                               temperature: float, time_step: float) -> Dict[str, int]:
        """
        Simulate a complete collision event including formation and decay.
        
        Args:
            initial_particles: Dictionary of initial particle counts
            temperature: Current collision temperature
            time_step: Time step in seconds
            
        Returns:
            Final particle counts after decay and formation
        """
        # Start with initial particles
        current_particles = initial_particles.copy()
        
        # Simulate decay of existing particles
        for particle_type, count in list(current_particles.items()):
            if count > 0:
                remaining, products = self.simulate_decay(particle_type, time_step, count)
                current_particles[particle_type] = remaining
                
                # Add decay products
                for product, product_count in products.items():
                    if product in current_particles:
                        current_particles[product] += product_count
                    else:
                        current_particles[product] = product_count
        
        # Simulate formation of new particles based on temperature
        for particle_type in ['deuteron', 'antideuteron']:
            formation_prob = self.calculate_formation_probability(temperature, particle_type)
            if formation_prob > 0:
                # Random number of new particles formed
                new_particles = np.random.poisson(formation_prob * 100)  # Average of 100 new particles
                if new_particles > 0:
                    if particle_type in current_particles:
                        current_particles[particle_type] += new_particles
                    else:
                        current_particles[particle_type] = new_particles
        
        return current_particles