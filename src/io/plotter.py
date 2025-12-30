import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

def plot_temperature_evolution(time_points: np.ndarray, 
                             temperature_points: np.ndarray,
                             save_plot: bool = False,
                             filename: str = "temperature_evolution.png"):
    """
    Plot the temperature evolution over time during the simulation.
    
    Args:
        time_points: Array of time points
        temperature_points: Array of temperature values
        save_plot: Whether to save the plot to file
        filename: Output filename if saving
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, temperature_points, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Evolution During Collision Event')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_particle_distribution(deuteron_counts: List[int],
                             antideuteron_counts: List[int],
                             time_points: List[float],
                             save_plot: bool = False,
                             filename: str = "particle_distribution.png"):
    """
    Plot the distribution of deuteron and antideuteron counts over time.
    
    Args:
        deuteron_counts: List of deuteron counts at each time step
        antideuteron_counts: List of antideuteron counts at each time step
        time_points: List of time points
        save_plot: Whether to save the plot to file
        filename: Output filename if saving
    """
    plt.figure(figsize=(12, 8))
    
    # Plot deuteron counts
    plt.subplot(2, 1, 1)
    plt.plot(time_points, deuteron_counts, 'g-', linewidth=2, label='Deuterons')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.title('Deuteron Production Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot antideuteron counts
    plt.subplot(2, 1, 2)
    plt.plot(time_points, antideuteron_counts, 'r-', linewidth=2, label='Antideuterons')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.title('Antideuteron Production Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_energy_threshold_analysis(threshold_energies: np.ndarray,
                                 formation_probabilities: np.ndarray,
                                 save_plot: bool = False,
                                 filename: str = "energy_threshold_analysis.png"):
    """
    Plot the relationship between energy thresholds and formation probabilities.
    
    Args:
        threshold_energies: Array of energy threshold values
        formation_probabilities: Array of corresponding formation probabilities
        save_plot: Whether to save the plot to file
        filename: Output filename if saving
    """
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_energies, formation_probabilities, 'o-', color='purple', linewidth=2)
    plt.xlabel('Energy Threshold (GeV)')
    plt.ylabel('Formation Probability')
    plt.title('Energy Threshold Analysis for Fragile Particle Formation')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_results(results: Dict, save_plot: bool = False):
    """
    Main visualization function that combines all plots.
    
    Args:
        results: Dictionary containing simulation results
        save_plot: Whether to save plots to files
    """
    # Temperature evolution plot
    if 'time_points' in results and 'temperature_points' in results:
        plot_temperature_evolution(
            results['time_points'],
            results['temperature_points'],
            save_plot,
            "temperature_evolution.png"
        )
    
    # Particle distribution plot
    if ('deuteron_counts' in results and 
        'antideuteron_counts' in results and 
        'time_points' in results):
        plot_particle_distribution(
            results['deuteron_counts'],
            results['antideuteron_counts'],
            results['time_points'],
            save_plot,
            "particle_distribution.png"
        )
    
    # Energy threshold analysis plot
    if ('threshold_energies' in results and 
        'formation_probabilities' in results):
        plot_energy_threshold_analysis(
            results['threshold_energies'],
            results['formation_probabilities'],
            save_plot,
            "energy_threshold_analysis.png"
        )