import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def plot_temperature_evolution(time_data: np.ndarray, temp_data: np.ndarray, 
                             save_plot: bool = False, filename: str = 'temperature_evolution.png'):
    """
    Plot the temperature evolution over time during the simulation.
    
    Args:
        time_data: Array of time values
        temp_data: Array of temperature values
        save_plot: Whether to save the plot to file
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, temp_data, 'b-', linewidth=2, label='Temperature')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Fireball Temperature Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_particle_yield(deuteron_counts: List[int], antideuteron_counts: List[int],
                       time_points: List[float], save_plot: bool = False,
                       filename: str = 'particle_yield.png'):
    """
    Plot deuteron and antideuteron yield over time.
    
    Args:
        deuteron_counts: List of deuteron counts at each time point
        antideuteron_counts: List of antideuteron counts at each time point
        time_points: List of time points
        save_plot: Whether to save the plot to file
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, deuteron_counts, 'b-', linewidth=2, label='Deuteron Yield')
    plt.plot(time_points, antideuteron_counts, 'r--', linewidth=2, label='Antideuteron Yield')
    plt.xlabel('Time (s)')
    plt.ylabel('Particle Count')
    plt.title('Deuteron and Antideuteron Production Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_energy_distribution(energies: np.ndarray, distribution: np.ndarray,
                           save_plot: bool = False, filename: str = 'energy_distribution.png'):
    """
    Plot the energy distribution of formed particles.
    
    Args:
        energies: Array of energy values
        distribution: Array of distribution values
        save_plot: Whether to save the plot to file
        filename: Name of the file to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energies, distribution, 'g-', linewidth=2)
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Probability Density')
    plt.title('Energy Distribution of Formed Particles')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_results(results: Dict, save_plot: bool = False):
    """
    Main visualization function that calls all plotting functions.
    
    Args:
        results: Dictionary containing simulation results
        save_plot: Whether to save plots to files
    """
    # Plot temperature evolution
    if 'time' in results and 'temperature' in results:
        plot_temperature_evolution(
            results['time'], 
            results['temperature'],
            save_plot,
            'temperature_evolution.png'
        )
    
    # Plot particle yield
    if 'deuteron_count' in results and 'antideuteron_count' in results:
        plot_particle_yield(
            results['deuteron_count'],
            results['antideuteron_count'],
            results.get('time', []),
            save_plot,
            'particle_yield.png'
        )
    
    # Plot energy distribution
    if 'energy_distribution' in results:
        plot_energy_distribution(
            results['energy_distribution']['energies'],
            results['energy_distribution']['distribution'],
            save_plot,
            'energy_distribution.png'
        )