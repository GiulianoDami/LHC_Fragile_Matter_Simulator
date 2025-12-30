import numpy as np
from scipy import integrate
from typing import Tuple, List

def calculate_temperature_evolution(initial_temp: float, time: np.ndarray, 
                                  cooling_rate: float = 1e12) -> np.ndarray:
    """
    Calculate temperature evolution over time during collision event.
    
    Args:
        initial_temp: Initial temperature in Kelvin
        time: Time array in seconds
        cooling_rate: Cooling rate constant
    
    Returns:
        Temperature array over time
    """
    return initial_temp * np.exp(-cooling_rate * time)

def calculate_energy_threshold(deuteron_binding_energy: float = 2.224, 
                             temperature: float = 1e12) -> float:
    """
    Calculate minimum energy required for deuteron formation.
    
    Args:
        deuteron_binding_energy: Binding energy of deuteron in MeV
        temperature: Current temperature in Kelvin
    
    Returns:
        Energy threshold in MeV
    """
    # Convert temperature to energy scale (kB*T)
    kb = 8.617e-5  # eV/K
    thermal_energy = kb * temperature
    
    # Return threshold energy (binding energy + thermal energy)
    return deuteron_binding_energy + thermal_energy

def integrate_particle_yield(energy_distribution: np.ndarray, 
                           cross_section: np.ndarray,
                           energy_grid: np.ndarray) -> float:
    """
    Integrate particle yield over energy distribution.
    
    Args:
        energy_distribution: Energy distribution function
        cross_section: Cross-section data
        energy_grid: Energy grid points
    
    Returns:
        Integrated particle yield
    """
    return integrate.simpson(energy_distribution * cross_section, energy_grid)

def calculate_decay_probability(lifetime: float, time: float) -> float:
    """
    Calculate probability of particle survival after given time.
    
    Args:
        lifetime: Particle lifetime in seconds
        time: Time elapsed in seconds
    
    Returns:
        Survival probability
    """
    return np.exp(-time / lifetime)

def generate_gaussian_noise(mean: float, std: float, size: int) -> np.ndarray:
    """
    Generate Gaussian distributed random numbers.
    
    Args:
        mean: Mean value
        std: Standard deviation
        size: Number of samples
    
    Returns:
        Array of Gaussian random numbers
    """
    return np.random.normal(mean, std, size)

def normalize_distribution(data: np.ndarray) -> np.ndarray:
    """
    Normalize a probability distribution to sum to 1.
    
    Args:
        data: Input data array
    
    Returns:
        Normalized distribution
    """
    return data / np.sum(data)

def calculate_statistical_uncertainty(counts: np.ndarray) -> np.ndarray:
    """
    Calculate statistical uncertainty for Poisson-distributed counts.
    
    Args:
        counts: Array of count values
    
    Returns:
        Array of uncertainties
    """
    return np.sqrt(counts)