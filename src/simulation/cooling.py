import numpy as np
from typing import Tuple, Dict, Any
import warnings

class CoolingFireball:
    """
    Models the cooling dynamics of a high-energy collision fireball,
    including temperature evolution and particle formation thresholds.
    """
    
    def __init__(self, 
                 initial_temp: float,
                 collision_energy: float,
                 duration: float,
                 volume: float = 1e-3,
                 expansion_rate: float = 1e20):
        """
        Initialize the cooling fireball model.
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature in Kelvin
        collision_energy : float
            Collision energy in GeV
        duration : float
            Total simulation time in seconds
        volume : float
            Initial fireball volume in fm^3
        expansion_rate : float
            Rate of fireball expansion in fm^3/s
        """
        self.initial_temp = initial_temp
        self.collision_energy = collision_energy
        self.duration = duration
        self.volume = volume
        self.expansion_rate = expansion_rate
        
        # Physical constants
        self.hbar = 6.582e-22  # GeV*s
        self.kb = 8.617e-5    # eV/K
        self.m_deuteron = 1.87561294257e9  # eV/c^2
        
        # Temperature thresholds for particle formation
        self.deuteron_threshold = 1.2e9  # eV
        self.antideuteron_threshold = 1.2e9  # eV
        
        # Time steps for simulation
        self.time_steps = int(1e5)
        self.dt = duration / self.time_steps
        
        # Initialize arrays for tracking evolution
        self.time_array = np.linspace(0, duration, self.time_steps)
        self.temp_array = np.zeros(self.time_steps)
        self.volume_array = np.zeros(self.time_steps)
        self.deuteron_density = np.zeros(self.time_steps)
        self.antideuteron_density = np.zeros(self.time_steps)
        
    def _calculate_temperature(self, t: float) -> float:
        """
        Calculate temperature at time t based on cooling law.
        Uses a simplified hydrodynamic cooling model.
        """
        # Simplified cooling law: T(t) = T_0 * (V_0/V(t))^(2/3)
        current_volume = self.volume * (1 + self.expansion_rate * t)
        temp = self.initial_temp * (self.volume / current_volume)**(2/3)
        
        # Apply additional cooling due to expansion
        temp *= np.exp(-t / (self.duration * 0.1))
        
        return max(temp, 1e6)  # Minimum temperature threshold
    
    def _calculate_density(self, temp: float) -> Tuple[float, float]:
        """
        Calculate deuteron and antideuteron densities based on temperature.
        
        Returns:
        --------
        tuple: (deuteron_density, antideuteron_density)
        """
        if temp < self.deuteron_threshold:
            return 0.0, 0.0
            
        # Simplified statistical calculation
        # Using Boltzmann factor for density estimation
        beta = 1.0 / (self.kb * temp)
        
        # Deuteron density (simplified)
        deuteron_density = 1e-15 * np.exp(-self.m_deuteron * beta)
        
        # Antideuteron density (same as deuteron for simplicity)
        antideuteron_density = deuteron_density
        
        return deuteron_density, antideuteron_density
    
    def simulate_cooling(self) -> Dict[str, np.ndarray]:
        """
        Run the complete cooling simulation.
        
        Returns:
        --------
        dict: Dictionary containing time evolution data
        """
        # Initialize arrays
        self.temp_array[0] = self.initial_temp
        self.volume_array[0] = self.volume
        
        # Main simulation loop
        for i in range(1, self.time_steps):
            t = self.time_array[i]
            
            # Update volume
            self.volume_array[i] = self.volume * (1 + self.expansion_rate * t)
            
            # Calculate new temperature
            self.temp_array[i] = self._calculate_temperature(t)
            
            # Calculate particle densities
            deuteron_dens, antideuteron_dens = self._calculate_density(
                self.temp_array[i]
            )
            
            self.deuteron_density[i] = deuteron_dens
            self.antideuteron_density[i] = antideuteron_dens
            
        return {
            'time': self.time_array,
            'temperature': self.temp_array,
            'volume': self.volume_array,
            'deuteron_density': self.deuteron_density,
            'antideuteron_density': self.antideuteron_density
        }
    
    def get_yield_statistics(self) -> Dict[str, float]:
        """
        Calculate final particle yields from the simulation.
        
        Returns:
        --------
        dict: Yield statistics for deuterons and antideuterons
        """
        # Find the time when temperature drops below formation threshold
        formation_times = np.where(
            self.temp_array >= self.deuteron_threshold
        )[0]
        
        if len(formation_times) == 0:
            return {'deuteron_count': 0, 'antideuteron_count': 0}
            
        # Use the last formation time for yield calculation
        last_formation_time = formation_times[-1]
        
        # Simple yield calculation based on density and volume
        deuteron_yield = (
            np.trapz(self.deuteron_density[:last_formation_time], 
                    self.time_array[:last_formation_time]) *
            self.volume
        )
        
        antideuteron_yield = (
            np.trapz(self.antideuteron_density[:last_formation_time], 
                    self.time_array[:last_formation_time]) *
            self.volume
        )
        
        return {
            'deuteron_count': max(0, deuteron_yield),
            'antideuteron_count': max(0, antideuteron_yield)
        }

def calculate_energy_thresholds() -> Dict[str, float]:
    """
    Calculate energy thresholds for deuteron and antideuteron formation.
    
    Returns:
    --------
    dict: Energy thresholds in eV
    """
    # Deuteron binding energy
    deuteron_binding = 2.224575e6  # eV
    
    # Threshold energies (approximate)
    thresholds = {
        'deuteron_formation': 1.2e9,  # eV
        'antideuteron_formation': 1.2e9,  # eV
        'binding_energy': deuteron_binding
    }
    
    return thresholds

def validate_parameters(initial_temp: float, collision_energy: float, 
                       duration: float) -> bool:
    """
    Validate input parameters for physical consistency.
    
    Parameters:
    -----------
    initial_temp : float
        Initial temperature in Kelvin
    collision_energy : float
        Collision energy in GeV
    duration : float
        Simulation duration in seconds
    
    Returns:
    --------
    bool: True if parameters are valid
    """
    if initial_temp <= 0 or collision_energy <= 0 or duration <= 0:
        warnings.warn("Non-positive parameters detected")
        return False
        
    if initial_temp < 1e9:
        warnings.warn("Initial temperature may be too low for deuteron formation")
        
    return True