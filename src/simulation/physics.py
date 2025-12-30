import numpy as np
from scipy import integrate
from typing import Dict, Tuple, List
import math

class PhysicsModel:
    """Core physics model for LHC fragile matter simulation"""
    
    def __init__(self):
        # Physical constants (SI units)
        self.kB = 1.380649e-23  # Boltzmann constant
        self.c = 299792458     # Speed of light
        self.hbar = 1.054571817e-34  # Reduced Planck constant
        self.m_p = 1.6726219e-27  # Proton mass
        self.m_n = 1.67492749804e-27  # Neutron mass
        self.m_d = 3.343583719e-27  # Deuteron mass
        self.m_antid = 3.343583719e-27  # Antideuteron mass
        
        # Energy thresholds
        self.deuteron_threshold = 2.224573e6 * 1.60218e-19  # Joules
        self.antideuteron_threshold = 2.224573e6 * 1.60218e-19  # Joules
        
        # Reaction cross-sections (approximate values)
        self.sigma_dd = 1e-28  # Deuteron-deuteron cross-section
        self.sigma_da = 1e-28  # Deuteron-antideuteron cross-section
        
    def calculate_temperature_evolution(self, initial_temp: float, 
                                     time: np.ndarray, 
                                     cooling_rate: float = 1e38) -> np.ndarray:
        """
        Calculate temperature evolution over time based on cooling dynamics
        
        Args:
            initial_temp: Initial temperature in Kelvin
            time: Array of time points
            cooling_rate: Rate of temperature decrease
            
        Returns:
            Array of temperatures at each time point
        """
        # Simplified exponential cooling model
        return initial_temp * np.exp(-cooling_rate * time)
    
    def calculate_deuteron_formation_probability(self, temperature: float) -> float:
        """
        Calculate probability of deuteron formation at given temperature
        
        Uses statistical mechanics approach based on binding energy
        """
        if temperature <= 0:
            return 0.0
            
        # Boltzmann factor for deuteron formation
        binding_energy = self.deuteron_threshold
        boltzmann_factor = np.exp(-binding_energy / (self.kB * temperature))
        
        # Formation probability (normalized)
        return min(1.0, boltzmann_factor)
    
    def calculate_antideuteron_formation_probability(self, temperature: float) -> float:
        """
        Calculate probability of antideuteron formation at given temperature
        """
        if temperature <= 0:
            return 0.0
            
        binding_energy = self.antideuteron_threshold
        boltzmann_factor = np.exp(-binding_energy / (self.kB * temperature))
        
        return min(1.0, boltzmann_factor)
    
    def calculate_reaction_rate(self, temperature: float, density: float) -> float:
        """
        Calculate reaction rate for particle formation
        
        Args:
            temperature: Current temperature in Kelvin
            density: Particle density
            
        Returns:
            Reaction rate coefficient
        """
        # Simplified reaction rate calculation using Maxwell-Boltzmann distribution
        thermal_velocity = np.sqrt(8 * self.kB * temperature / (np.pi * self.m_p))
        return density * thermal_velocity * self.sigma_dd
    
    def simulate_decay_chain(self, initial_particles: Dict[str, int], 
                           time_step: float, total_time: float) -> Dict[str, List[int]]:
        """
        Simulate particle decay chain over time
        
        Args:
            initial_particles: Dictionary of initial particle counts
            time_step: Time step for simulation
            total_time: Total simulation time
            
        Returns:
            Dictionary of particle counts over time
        """
        time_points = np.arange(0, total_time, time_step)
        results = {key: [value] for key, value in initial_particles.items()}
        
        # Decay constants (approximate)
        decay_constants = {
            'deuteron': 1e-22,
            'antideuteron': 1e-22
        }
        
        for t in time_points[1:]:
            new_counts = {}
            for particle_type, count in results.items():
                if particle_type in decay_constants:
                    # Simple exponential decay
                    decay_rate = decay_constants[particle_type]
                    new_count = max(0, count[-1] * np.exp(-decay_rate * time_step))
                    new_counts[particle_type] = new_count
                else:
                    new_counts[particle_type] = count[-1]
            
            # Update results
            for key, value in new_counts.items():
                results[key].append(value)
                
        return results
    
    def calculate_energy_threshold(self, particle_mass: float, 
                                 temperature: float) -> float:
        """
        Calculate minimum energy required for particle formation
        
        Args:
            particle_mass: Mass of particle in kg
            temperature: Current temperature in Kelvin
            
        Returns:
            Minimum energy threshold in Joules
        """
        # Thermal energy threshold
        thermal_energy = 3 * self.kB * temperature / 2
        # Binding energy requirement
        binding_energy = particle_mass * self.c**2
        
        return thermal_energy + binding_energy
    
    def compute_yield_distribution(self, temperature: float, 
                                num_events: int = 1000) -> Dict[str, float]:
        """
        Compute statistical distribution of particle yields
        
        Args:
            temperature: Current temperature in Kelvin
            num_events: Number of Monte Carlo events
            
        Returns:
            Dictionary of average yields
        """
        deuteron_probs = []
        antideuteron_probs = []
        
        for _ in range(num_events):
            # Random temperature fluctuation
            temp_fluct = np.random.normal(temperature, temperature * 0.1)
            
            deuteron_prob = self.calculate_deuteron_formation_probability(temp_fluct)
            antideuteron_prob = self.calculate_antideuteron_formation_probability(temp_fluct)
            
            deuteron_probs.append(deuteron_prob)
            antideuteron_probs.append(antideuteron_prob)
        
        return {
            'deuteron_yield': np.mean(deuteron_probs),
            'antideuteron_yield': np.mean(antideuteron_probs)
        }

# Utility functions for physics calculations
def calculate_boltzmann_factor(energy: float, temperature: float) -> float:
    """Calculate Boltzmann factor for given energy and temperature"""
    k_B = 1.380649e-23  # Boltzmann constant
    return np.exp(-energy / (k_B * temperature))

def calculate_maxwell_boltzmann_velocity(temperature: float, mass: float, 
                                       dimension: int = 3) -> float:
    """Calculate most probable velocity from Maxwell-Boltzmann distribution"""
    k_B = 1.380649e-23
    return np.sqrt(dimension * k_B * temperature / mass)

def calculate_cross_section_ratio(sigma1: float, sigma2: float) -> float:
    """Calculate ratio of two cross-sections"""
    return sigma1 / sigma2 if sigma2 != 0 else 0.0