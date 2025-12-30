import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

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
        self.hbar = 6.582e-22  # Reduced Planck constant in GeV*s
        self.kb = 8.617e-5    # Boltzmann constant in eV/K
        self.deuteron_mass = 1.87561294e9  # Deuteron mass in eV/c^2
        self.antideuteron_mass = 1.87561294e9  # Antideuteron mass in eV/c^2
        
        # Simulation parameters
        self.time_steps = 1000
        self.temperature_history = []
        self.particle_counts = {'deuteron': 0, 'antideuteron': 0}
        
    def _calculate_temperature_evolution(self) -> np.ndarray:
        """
        Calculate temperature evolution over time using a simplified cooling model.
        
        Returns:
            Array of temperatures at each time step
        """
        # Simplified cooling law: T(t) = T0 * exp(-t/tau)
        tau = self.duration / 5  # Cooling timescale
        t = np.linspace(0, self.duration, self.time_steps)
        temperatures = self.initial_temperature * np.exp(-t / tau)
        return temperatures
    
    def _calculate_energy_threshold(self) -> float:
        """
        Calculate the minimum energy required for deuteron formation.
        
        Returns:
            Energy threshold in GeV
        """
        # Deuteron binding energy ~2.224 MeV
        binding_energy = 2.224e-3  # Convert to GeV
        return binding_energy
    
    def _simulate_particle_formation(self, temperature: float) -> Tuple[int, int]:
        """
        Simulate deuteron and antideuteron formation at given temperature.
        
        Args:
            temperature: Current temperature in Kelvin
            
        Returns:
            Tuple of (deuteron_count, antideuteron_count)
        """
        # Convert temperature to GeV scale
        temp_gev = temperature * self.kb / 1e9  # Approximate conversion
        
        # Energy threshold for formation
        threshold = self._calculate_energy_threshold()
        
        # Formation probability based on temperature and threshold
        if temp_gev > threshold:
            # Simple statistical model - higher temperature = more particles
            formation_rate = min(1.0, temp_gev / threshold)
            deuteron_count = int(np.random.poisson(formation_rate * 100))
            antideuteron_count = int(np.random.poisson(formation_rate * 50))  # Fewer antideuterons
        else:
            deuteron_count = 0
            antideuteron_count = 0
            
        return deuteron_count, antideuteron_count
    
    def run_simulation(self) -> Dict:
        """
        Run the complete particle formation simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        logger.info("Starting particle formation simulation")
        
        # Calculate temperature evolution
        temperatures = self._calculate_temperature_evolution()
        self.temperature_history = temperatures.tolist()
        
        # Simulate particle formation at each time step
        total_deuterons = 0
        total_antideuterons = 0
        
        for i, temp in enumerate(temperatures):
            deuteron_count, antideuteron_count = self._simulate_particle_formation(temp)
            total_deuterons += deuteron_count
            total_antideuterons += antideuteron_count
            
            # Store counts at specific intervals for monitoring
            if i % (self.time_steps // 10) == 0:
                logger.debug(f"Time step {i}: T={temp:.2e}K, D={deuteron_count}, Dbar={antideuteron_count}")
        
        # Store final results
        self.particle_counts['deuteron'] = total_deuterons
        self.particle_counts['antideuteron'] = total_antideuterons
        
        results = {
            'deuteron_count': total_deuterons,
            'antideuteron_count': total_antideuterons,
            'temperature_history': self.temperature_history,
            'initial_temperature': self.initial_temperature,
            'collision_energy': self.collision_energy,
            'duration': self.duration
        }
        
        logger.info(f"Simulation completed. Deuterons: {total_deuterons}, Antideuterons: {total_antideuterons}")
        return results