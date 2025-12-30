import numpy as np
from typing import Dict, Tuple
import logging

class ParticleSimulator:
    def __init__(self, initial_temperature: float, collision_energy: float, duration: float):
        """
        Initialize the particle simulator with collision parameters.
        
        Args:
            initial_temperature: Initial temperature of the fireball in Kelvin
            collision_energy: Collision energy in GeV
            duration: Simulation duration in seconds
        """
        self.initial_temperature = initial_temperature
        self.collision_energy = collision_energy
        self.duration = duration
        
        # Physical constants
        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.m_deuteron = 1.87561294257e9  # Deuteron mass in eV/c^2
        self.m_antideuteron = 1.87561294257e9  # Antideuteron mass in eV/c^2
        self.energy_threshold = 2 * self.m_deuteron  # Minimum energy for deuteron formation
        
        # Simulation parameters
        self.time_steps = 1000
        self.dt = duration / self.time_steps
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _temperature_evolution(self, t: float) -> float:
        """
        Calculate temperature at time t based on cooling fireball model.
        
        Args:
            t: Time in seconds
            
        Returns:
            Temperature in Kelvin
        """
        # Simplified cooling law: T(t) = T0 * exp(-kt)
        # k is a cooling constant
        k = 1e25  # Cooling rate constant (arbitrary units)
        return self.initial_temperature * np.exp(-k * t)
    
    def _calculate_energy_density(self, temperature: float) -> float:
        """
        Calculate energy density based on temperature.
        
        Args:
            temperature: Current temperature in Kelvin
            
        Returns:
            Energy density in GeV/fm^3
        """
        # Stefan-Boltzmann law scaled for nuclear matter
        sigma = 5.67e-15  # Stefan-Boltzmann constant
        energy_density = (sigma * temperature**4) / (np.pi**2 * 1e3)  # Convert to GeV/fm^3
        return energy_density
    
    def _calculate_formation_probability(self, temperature: float) -> float:
        """
        Calculate probability of deuteron formation at given temperature.
        
        Args:
            temperature: Current temperature in Kelvin
            
        Returns:
            Formation probability (0-1)
        """
        # Simple exponential dependence on temperature
        if temperature < 1e11:
            return 0.0
            
        # Probability increases with temperature above threshold
        threshold_temp = 1e11
        prob = np.exp((temperature - threshold_temp) / (1e11))
        return min(prob, 1.0)
    
    def _simulate_decay_chain(self, temperature: float) -> Tuple[int, int]:
        """
        Simulate particle decay chain to produce deuterons and antideuterons.
        
        Args:
            temperature: Current temperature in Kelvin
            
        Returns:
            Tuple of (deuteron_count, antideuteron_count)
        """
        # Simplified statistical model
        formation_prob = self._calculate_formation_probability(temperature)
        
        # Number of nucleon pairs available (simplified)
        nucleon_pairs = max(0, int(1000 * formation_prob * np.random.rand()))
        
        # Production rates (simplified)
        deuteron_rate = 0.3
        antideuteron_rate = 0.2
        
        deuteron_count = int(nucleon_pairs * deuteron_rate * np.random.rand())
        antideuteron_count = int(nucleon_pairs * antideuteron_rate * np.random.rand())
        
        return deuteron_count, antideuteron_count
    
    def run_simulation(self) -> Dict:
        """
        Run the complete particle formation simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info("Starting particle formation simulation")
        
        # Initialize tracking variables
        time_points = []
        temp_points = []
        deuteron_counts = []
        antideuteron_counts = []
        energy_densities = []
        
        total_deuteron_count = 0
        total_antideuteron_count = 0
        
        # Main simulation loop
        for i in range(self.time_steps):
            t = i * self.dt
            temperature = self._temperature_evolution(t)
            energy_density = self._calculate_energy_density(temperature)
            
            # Track values for analysis
            time_points.append(t)
            temp_points.append(temperature)
            energy_densities.append(energy_density)
            
            # Simulate particle formation
            deuteron_count, antideuteron_count = self._simulate_decay_chain(temperature)
            
            deuteron_counts.append(deuteron_count)
            antideuteron_counts.append(antideuteron_count)
            
            total_deuteron_count += deuteron_count
            total_antideuteron_count += antideuteron_count
            
            # Log progress every 100 steps
            if i % 100 == 0:
                self.logger.debug(f"Time: {t:.2e}s, Temp: {temperature:.2e}K, "
                                f"Deuterons: {deuteron_count}, Antideuterons: {antideuteron_count}")
        
        # Prepare results dictionary
        results = {
            'time_points': time_points,
            'temperature_points': temp_points,
            'deuteron_counts': deuteron_counts,
            'antideuteron_counts': antideuteron_counts,
            'energy_densities': energy_densities,
            'deuteron_count': total_deuteron_count,
            'antideuteron_count': total_antideuteron_count,
            'final_temperature': temperature,
            'initial_temperature': self.initial_temperature
        }
        
        self.logger.info(f"Simulation completed. Final deuteron count: {total_deuteron_count}")
        return results