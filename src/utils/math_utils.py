import numpy as np
from scipy import integrate
from typing import Tuple, List

def calculate_temperature_evolution(initial_temp: float, time: np.ndarray, 
                                  cooling_rate: float = 1e38) -> np.ndarray:
    """
    Calculate temperature evolution over time based on cooling rate.
    
    Args:
        initial_temp: Initial temperature in Kelvin
        time: Array of time points
        cooling_rate: Rate of temperature decrease (default: 1e38 K/s)
        
    Returns:
        Array of temperatures at each time point
    """
    return initial_temp * np.exp(-cooling_rate * time)

def calculate_energy_threshold(deuteron_mass: float = 1.875612942e-24, 
                             binding_energy: float = 2.224575e-12) -> float:
    """
    Calculate minimum energy required for deuteron formation.
    
    Args:
        deuteron_mass: Mass of deuteron in kg
        binding_energy: Binding energy in Joules
        
    Returns:
        Minimum energy threshold in Joules
    """
    return deuteron_mass * 299792458**2 + binding_energy

def integrate_particle_yield(temperature_profile: np.ndarray, 
                           time_points: np.ndarray,
                           formation_cross_section: callable,
                           particle_density: float = 1e30) -> float:
    """
    Integrate particle yield over time using numerical integration.
    
    Args:
        temperature_profile: Array of temperatures over time
        time_points: Time points corresponding to temperature profile
        formation_cross_section: Function returning cross-section at given temp
        particle_density: Number density of target particles
        
    Returns:
        Total integrated particle yield
    """
    # Calculate formation rate at each time step
    formation_rates = []
    for T in temperature_profile:
        sigma = formation_cross_section(T)
        rate = sigma * particle_density * 299792458  # Simplified rate calculation
        formation_rates.append(rate)
    
    formation_rates = np.array(formation_rates)
    
    # Integrate using trapezoidal rule
    yield_integral = integrate.trapz(formation_rates, time_points)
    
    return yield_integral

def boltzmann_distribution(energy: float, temperature: float) -> float:
    """
    Calculate Boltzmann probability distribution.
    
    Args:
        energy: Energy in Joules
        temperature: Temperature in Kelvin
        
    Returns:
        Probability density
    """
    k_boltzmann = 1.380649e-23  # J/K
    if temperature <= 0:
        return 0
    return np.exp(-energy / (k_boltzmann * temperature))

def calculate_decay_chain_probability(decay_constants: List[float], 
                                    time: float) -> float:
    """
    Calculate probability of surviving decay chain.
    
    Args:
        decay_constants: List of decay constants for each stage
        time: Time elapsed
        
    Returns:
        Survival probability
    """
    total_decay = sum([k * time for k in decay_constants])
    return np.exp(-total_decay)

def statistical_sampling(n_samples: int, distribution_func: callable, 
                        param_range: Tuple[float, float]) -> np.ndarray:
    """
    Generate statistical samples from a given distribution.
    
    Args:
        n_samples: Number of samples to generate
        distribution_func: Function defining the distribution
        param_range: Range of parameters for sampling
        
    Returns:
        Array of sampled values
    """
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(param_range[0], param_range[1])
        y = np.random.uniform(0, 1)
        if y <= distribution_func(x):
            samples.append(x)
    
    return np.array(samples)