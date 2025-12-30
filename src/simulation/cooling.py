import numpy as np
from typing import Tuple, Dict, Any

class CoolingFireball:
    """
    Models the cooling dynamics of a high-energy collision fireball
    following the principles of relativistic hydrodynamics and thermalization.
    """
    
    def __init__(self, initial_temp: float, collision_energy: float, 
                 duration: float, time_step: float = 1e-24):
        """
        Initialize the cooling fireball model.
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature of the fireball in Kelvin
        collision_energy : float
            Collision energy in GeV
        duration : float
            Total simulation duration in seconds
        time_step : float
            Time step for numerical integration in seconds
        """
        self.initial_temp = initial_temp
        self.collision_energy = collision_energy
        self.duration = duration
        self.time_step = time_step
        
        # Physical constants
        self.hbar = 1.054571817e-34  # J⋅s
        self.kB = 1.380649e-23       # J/K
        self.c = 299792458           # m/s
        self.m_p = 1.6726219e-27     # kg (proton mass)
        
        # Cooling parameters
        self.gamma = 4/3  # Adiabatic index for ideal gas
        self.alpha = 0.1  # Cooling rate parameter
        
        # Initialize arrays
        self.time_points = np.arange(0, duration + time_step, time_step)
        self.temperature_history = np.zeros_like(self.time_points)
        self.temperature_history[0] = initial_temp
        
        # Particle statistics
        self.deuteron_count = 0
        self.antideuteron_count = 0
        
    def calculate_cooling_rate(self, temp: float) -> float:
        """
        Calculate the cooling rate based on temperature and collision energy.
        
        Parameters:
        -----------
        temp : float
            Current temperature in Kelvin
            
        Returns:
        --------
        float
            Cooling rate coefficient
        """
        # Simplified cooling model based on energy density and expansion
        energy_density = (temp**4) * (np.pi**2 / 15) * (self.kB**4 / self.hbar**3)
        expansion_rate = 1e20  # Simplified expansion rate
        
        return self.alpha * energy_density / expansion_rate
    
    def evolve_temperature(self) -> None:
        """
        Evolve the temperature through time using numerical integration.
        """
        for i in range(1, len(self.time_points)):
            dt = self.time_step
            current_temp = self.temperature_history[i-1]
            
            # Simplified cooling equation: dT/dt = -cooling_rate * T
            cooling_rate = self.calculate_cooling_rate(current_temp)
            dT_dt = -cooling_rate * current_temp
            
            # Euler integration
            new_temp = current_temp + dT_dt * dt
            
            # Ensure temperature doesn't go below absolute zero
            self.temperature_history[i] = max(new_temp, 0)
    
    def calculate_formation_probability(self, temp: float) -> Tuple[float, float]:
        """
        Calculate the probability of deuteron and antideuteron formation.
        
        Parameters:
        -----------
        temp : float
            Current temperature in Kelvin
            
        Returns:
        --------
        tuple(float, float)
            Probability of deuteron and antideuteron formation respectively
        """
        # Formation threshold temperature (approximate)
        threshold_temp = 1e9  # Kelvin
        
        if temp < threshold_temp:
            return 0.0, 0.0
            
        # Boltzmann factor for formation
        boltzmann_factor = np.exp(-1.5e9 / temp)  # Simplified binding energy effect
        
        # Deuteron formation probability (simplified)
        deuteron_prob = boltzmann_factor * (temp / threshold_temp)**(-2)
        
        # Antideuteron probability (same but includes charge conjugation)
        antideuteron_prob = deuteron_prob * 0.9  # Slight asymmetry
        
        return min(deuteron_prob, 1.0), min(antideuteron_prob, 1.0)
    
    def simulate_formation(self) -> None:
        """
        Simulate particle formation throughout the cooling process.
        """
        for i, temp in enumerate(self.temperature_history):
            deuteron_prob, antideuteron_prob = self.calculate_formation_probability(temp)
            
            # Stochastic formation process
            if np.random.random() < deuteron_prob:
                self.deuteron_count += 1
                
            if np.random.random() < antideuteron_prob:
                self.antideuteron_count += 1
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Execute the full cooling simulation.
        
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        self.evolve_temperature()
        self.simulate_formation()
        
        return {
            'temperature_history': self.temperature_history,
            'time_points': self.time_points,
            'deuteron_count': self.deuteron_count,
            'antideuteron_count': self.antideuteron_count,
            'final_temperature': self.temperature_history[-1],
            'initial_temperature': self.initial_temp
        }