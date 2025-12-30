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
        self.m_p = 1.6726219e-27  # Proton mass
        self.m_n = 1.6749275e-27  # Neutron mass
        self.m_d = 3.3435837e-27  # Deuteron mass
        self.m_antid = 3.3435837e-27  # Antideuteron mass
        self.hbar = 1.0545718e-34  # Reduced Planck constant
        
        # Energy thresholds
        self.deuteron_threshold = 2.224573e6 * 1.60218e-19  # Joules
        self.antideuteron_threshold = 2.224573e6 * 1.60218e-19  # Joules
        
        # Reaction cross-sections (approximate values)
        self.sigma_dd = 1e-28  # Deuteron-deuteron cross-section
        self.sigma_da = 1e-28  # Deuteron-antideuteron cross-section
        self.sigma_pp = 1e-28  # Proton-proton cross-section
    
    def temperature_evolution(self, t: float, T0: float, tau: float) -> float:
        """
        Calculate temperature evolution over time following a cooling fireball model
        
        Args:
            t: Time in seconds
            T0: Initial temperature (K)
            tau: Cooling timescale (s)
            
        Returns:
            Temperature at time t (K)
        """
        return T0 * np.exp(-t / tau)
    
    def calculate_deuteron_yield(self, T: float, density: float) -> float:
        """
        Calculate deuteron formation probability based on temperature and density
        
        Args:
            T: Temperature (K)
            density: Particle density (m^-3)
            
        Returns:
            Deuteron formation rate (per second)
        """
        if T < 1e9:
            return 0.0
            
        # Boltzmann factor for deuteron formation
        boltzmann_factor = np.exp(-self.deuteron_threshold / (self.kB * T))
        
        # Formation rate proportional to density squared and Boltzmann factor
        formation_rate = density**2 * boltzmann_factor * 1e-20
        
        return formation_rate
    
    def calculate_antideuteron_yield(self, T: float, density: float) -> float:
        """
        Calculate antideuteron formation probability based on temperature and density
        
        Args:
            T: Temperature (K)
            density: Particle density (m^-3)
            
        Returns:
            Antideuteron formation rate (per second)
        """
        if T < 1e9:
            return 0.0
            
        # Boltzmann factor for antideuteron formation
        boltzmann_factor = np.exp(-self.antideuteron_threshold / (self.kB * T))
        
        # Formation rate proportional to density squared and Boltzmann factor
        formation_rate = density**2 * boltzmann_factor * 1e-20
        
        return formation_rate
    
    def calculate_reaction_rate(self, T: float, species: str = 'pp') -> float:
        """
        Calculate reaction rate based on temperature using Arrhenius-like formula
        
        Args:
            T: Temperature (K)
            species: Type of reaction ('pp', 'dd', 'da')
            
        Returns:
            Reaction rate coefficient (m^3/s)
        """
        # Activation energies for different reactions (eV)
        activation_energies = {
            'pp': 1.0e6,
            'dd': 2.2e6,
            'da': 2.2e6
        }
        
        Ea = activation_energies.get(species, 1.0e6)
        Ea_J = Ea * 1.60218e-19  # Convert eV to Joules
        
        # Pre-exponential factor (approximate)
        pre_exp = 1e-10
        
        # Arrhenius equation
        rate = pre_exp * np.exp(-Ea_J / (self.kB * T))
        
        return rate
    
    def calculate_density(self, T: float, volume: float) -> float:
        """
        Estimate particle density from temperature and volume
        
        Args:
            T: Temperature (K)
            volume: Volume of system (m^3)
            
        Returns:
            Particle density (m^-3)
        """
        # Ideal gas law approximation
        # Using proton mass as representative particle mass
        n = (1e28 * T / (self.kB * T)) / volume  # Simplified estimate
        
        return max(n, 1e10)  # Minimum density
    
    def simulate_decay_chain(self, initial_particles: Dict[str, int], 
                           time_step: float, total_time: float) -> Dict[str, List[int]]:
        """
        Simulate particle decay chain over time
        
        Args:
            initial_particles: Dictionary of initial particle counts
            time_step: Time step for simulation (s)
            total_time: Total simulation time (s)
            
        Returns:
            Dictionary of particle counts over time
        """
        # Decay constants (approximate half-lives in seconds)
        decay_constants = {
            'deuteron': 1e-21,  # ~100 ps
            'antideuteron': 1e-21  # ~100 ps
        }
        
        # Initialize tracking
        time_points = np.arange(0, total_time, time_step)
        particle_history = {key: [value] for key, value in initial_particles.items()}
        
        # Simple exponential decay model
        for i, t in enumerate(time_points[1:], 1):
            for particle_type in ['deuteron', 'antideuteron']:
                if particle_type in initial_particles:
                    current_count = particle_history[particle_type][-1]
                    decayed = current_count * (1 - np.exp(-decay_constants[particle_type] * time_step))
                    new_count = max(0, current_count - decayed)
                    particle_history[particle_type].append(new_count)
        
        return particle_history
    
    def calculate_energy_threshold(self, particle_mass: float, T: float) -> float:
        """
        Calculate minimum energy required for particle formation
        
        Args:
            particle_mass: Mass of particle (kg)
            T: Temperature (K)
            
        Returns:
            Minimum energy threshold (J)
        """
        # Thermal energy per degree of freedom
        thermal_energy = 3 * self.kB * T / 2
        
        # Required binding energy plus thermal energy
        binding_energy = particle_mass * self.c**2  # Rest mass energy
        
        return binding_energy + thermal_energy
    
    def compute_collision_cross_section(self, energy: float, particle_type: str) -> float:
        """
        Compute collision cross-section based on energy and particle type
        
        Args:
            energy: Collision energy (GeV)
            particle_type: Type of particles involved
            
        Returns:
            Cross-section in m^2
        """
        # Simplified power-law dependence
        if particle_type == 'proton':
            return self.sigma_pp * (energy / 1000)**(-0.5)
        elif particle_type == 'deuteron':
            return self.sigma_dd * (energy / 1000)**(-0.3)
        elif particle_type == 'antideuteron':
            return self.sigma_da * (energy / 1000)**(-0.3)
        else:
            return 1e-28