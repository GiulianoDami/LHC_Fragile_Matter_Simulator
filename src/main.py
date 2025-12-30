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
    A Monte Carlo simulation framework for modeling fragile nuclear particle formation
    in high-energy collision environments.
    """
    
    def __init__(self, initial_temperature: float, collision_energy: float, duration: float):
        """
        Initialize the particle simulator with collision parameters.
        
        Args:
            initial_temperature (float): Initial temperature in Kelvin
            collision_energy (float): Collision energy in GeV
            duration (float): Simulation duration in seconds
        """
        self.initial_temperature = initial_temperature
        self.collision_energy = collision_energy
        self.duration = duration
        
        # Physical constants
        self.kb = 8.617e-5  # Boltzmann constant in eV/K
        self.m_deuteron = 1.875612942e9  # Deuteron mass in eV/c^2
        self.m_antideuteron = 1.875612942e9  # Antideuteron mass in eV/c^2
        self.energy_threshold = 2 * self.m_deuteron  # Energy threshold for deuteron formation
        
        # Simulation parameters
        self.deuteron_count = 0
        self.antideuteron_count = 0
        self.temperature_history = []
        self.time_history = []
        
    def temperature_evolution(self, t: float, T: float) -> float:
        """
        Model the temperature evolution during the collision event.
        
        Args:
            t (float): Time in seconds
            T (float): Current temperature in Kelvin
            
        Returns:
            float: Rate of temperature change
        """
        # Simplified cooling model: exponential decay with time constant
        tau = 1e-23  # Cooling time constant in seconds
        return -T / tau
    
    def calculate_formation_probability(self, temperature: float) -> Tuple[float, float]:
        """
        Calculate the probability of deuteron and antideuteron formation.
        
        Args:
            temperature (float): Current temperature in Kelvin
            
        Returns:
            Tuple[float, float]: Probability of deuteron and antideuteron formation
        """
        # Convert temperature to energy scale
        energy_scale = temperature * self.kb
        
        # Formation probability based on energy threshold
        if energy_scale > self.energy_threshold:
            # Simplified statistical model
            prob_deuteron = min(1.0, np.exp(-self.energy_threshold / (energy_scale)))
            prob_antideuteron = min(1.0, np.exp(-self.energy_threshold / (energy_scale)))
        else:
            prob_deuteron = 0.0
            prob_antideuteron = 0.0
            
        return prob_deuteron, prob_antideuteron
    
    def run_simulation(self) -> Dict[str, float]:
        """
        Run the complete simulation.
        
        Returns:
            Dict[str, float]: Simulation results including particle counts
        """
        logger.info("Starting particle formation simulation...")
        
        # Initial conditions
        t_span = (0, self.duration)
        y0 = [self.initial_temperature]
        
        # Solve ODE for temperature evolution
        sol = solve_ivp(self.temperature_evolution, t_span, y0, 
                       method='RK45', t_eval=np.linspace(0, self.duration, 1000))
        
        # Extract temperature history
        self.temperature_history = sol.y[0]
        self.time_history = sol.t
        
        # Monte Carlo particle formation process
        num_steps = len(self.time_history)
        for i in range(num_steps):
            current_temp = self.temperature_history[i]
            
            # Calculate formation probabilities
            prob_deuteron, prob_antideuteron = self.calculate_formation_probability(current_temp)
            
            # Random sampling for particle formation
            if np.random.random() < prob_deuteron:
                self.deuteron_count += 1
                
            if np.random.random() < prob_antideuteron:
                self.antideuteron_count += 1
                
        logger.info(f"Simulation completed. Deuteron count: {self.deuteron_count}, "
                   f"Antideuteron count: {self.antideuteron_count}")
        
        return {
            'deuteron_count': self.deuteron_count,
            'antideuteron_count': self.antideuteron_count,
            'final_temperature': self.temperature_history[-1],
            'temperature_history': self.temperature_history,
            'time_history': self.time_history
        }

def visualize_results(results: Dict, save_plot: bool = False):
    """
    Visualize the simulation results.
    
    Args:
        results (Dict): Results from the simulation
        save_plot (bool): Whether to save the plot to file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot temperature evolution
    ax1.plot(results['time_history'], results['temperature_history'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Temperature Evolution During Collision')
    ax1.grid(True)
    
    # Plot particle counts
    ax2.bar(['Deuteron', 'Antideuteron'], 
            [results['deuteron_count'], results['antideuteron_count']])
    ax2.set_ylabel('Particle Count')
    ax2.set_title('Final Particle Distribution')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('simulation_results.png')
        logger.info("Plot saved as 'simulation_results.png'")
    
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
    
    visualize_results(results)