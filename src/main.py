import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParticleSimulator:
    """
    A Monte Carlo simulation framework for modeling fragile nuclear particle 
    formation in high-energy collision environments.
    """
    
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
        self.m_proton = 938.272  # Proton mass in MeV/c^2
        self.m_neutron = 939.565  # Neutron mass in MeV/c^2
        self.m_deuteron = 1875.613  # Deuteron mass in MeV/c^2
        self.m_antideuteron = 1875.613  # Antideuteron mass in MeV/c^2
        
        # Energy thresholds
        self.deuteron_threshold = 2 * self.m_proton  # Minimum energy for deuteron formation
        self.antideuteron_threshold = 2 * self.m_proton  # Minimum energy for antideuteron formation
        
        # Simulation parameters
        self.time_points = np.linspace(0, duration, 1000)
        
    def temperature_evolution(self, t: float, T: float) -> float:
        """
        Model the temperature evolution of the fireball.
        
        Args:
            t: Time in seconds
            T: Current temperature in Kelvin
            
        Returns:
            Rate of temperature change dT/dt
        """
        # Cooling rate proportional to temperature squared
        cooling_rate = -0.1 * T**2 / (1 + 0.01 * T)
        return cooling_rate
    
    def calculate_energy_density(self, temperature: float) -> float:
        """
        Calculate energy density based on temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Energy density in GeV/fm^3
        """
        # Simplified energy density calculation
        # Using Stefan-Boltzmann law scaled appropriately
        energy_density = 0.1 * (temperature / 1e12)**4  # Approximation
        return energy_density
    
    def calculate_deuteron_yield(self, temperature: float) -> float:
        """
        Calculate deuteron production probability based on temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Probability of deuteron formation
        """
        # Simple model: deuteron formation becomes possible above threshold
        if temperature > 1e11:
            # Probability increases with temperature above threshold
            prob = min(1.0, (temperature - 1e11) / 1e11)
            return prob
        return 0.0
    
    def calculate_antideuteron_yield(self, temperature: float) -> float:
        """
        Calculate antideuteron production probability based on temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Probability of antideuteron formation
        """
        # Simple model: antideuteron formation becomes possible above threshold
        if temperature > 1e11:
            # Probability increases with temperature above threshold
            prob = min(1.0, (temperature - 1e11) / 1e11)
            return prob
        return 0.0
    
    def run_simulation(self) -> Dict:
        """
        Run the complete simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        logger.info("Starting simulation...")
        
        # Solve temperature evolution ODE
        sol = solve_ivp(
            fun=lambda t, T: [self.temperature_evolution(t, T[0])],
            t_span=(0, self.duration),
            y0=[self.initial_temperature],
            t_eval=self.time_points,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        temperatures = sol.y[0]
        
        # Calculate yields over time
        deuteron_counts = []
        antideuteron_counts = []
        
        for temp in temperatures:
            deuteron_prob = self.calculate_deuteron_yield(temp)
            antideuteron_prob = self.calculate_antideuteron_yield(temp)
            
            # Convert probabilities to counts (simplified)
            deuteron_count = int(deuteron_prob * 1000)
            antideuteron_count = int(antideuteron_prob * 1000)
            
            deuteron_counts.append(deuteron_count)
            antideuteron_counts.append(antideuteron_count)
        
        # Aggregate final results
        total_deuteron_count = sum(deuteron_counts)
        total_antideuteron_count = sum(antideuteron_counts)
        
        results = {
            'temperatures': temperatures,
            'time_points': self.time_points,
            'deuteron_count': total_deuteron_count,
            'antideuteron_count': total_antideuteron_count,
            'final_temperature': temperatures[-1],
            'energy_density': self.calculate_energy_density(temperatures[-1])
        }
        
        logger.info(f"Simulation completed. Deuteron count: {total_deuteron_count}, "
                   f"Antideuteron count: {total_antideuteron_count}")
        
        return results

def visualize_results(results: Dict, save_plot: bool = False):
    """
    Visualize simulation results.
    
    Args:
        results: Dictionary containing simulation results
        save_plot: Whether to save the plot to file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot temperature evolution
    ax1.plot(results['time_points'], results['temperatures'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Fireball Temperature Evolution')
    ax1.grid(True)
    
    # Plot particle counts
    ax2.plot(results['time_points'], [results['deuteron_count']] * len(results['time_points']), 
             'g--', label='Deuteron Count')
    ax2.plot(results['time_points'], [results['antideuteron_count']] * len(results['time_points']), 
             'r--', label='Antideuteron Count')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Particle Count')
    ax2.set_title('Particle Production Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('simulation_results.png')
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    sim = ParticleSimulator(
        initial_temperature=1.5e12,
        collision_energy=13000,
        duration=1e-23
    )
    
    results = sim.run_simulation()
    
    print(f"Deuteron yield: {results['deuteron_count']}")
    print(f"Antideuteron yield: {results['antideuteron_count']}")
    print(f"Final temperature: {results['final_temperature']:.2e} K")
    print(f"Energy density: {results['energy_density']:.2f} GeV/fm³")
    
    # Uncomment to generate plots
    # visualize_results(results, save_plot=True)