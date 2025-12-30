import numpy as np
from typing import Dict, List, Tuple
import random

class Particle:
    """Represents a fundamental particle in the simulation."""
    
    def __init__(self, name: str, mass: float, charge: int, is_antiparticle: bool = False):
        self.name = name
        self.mass = mass  # in GeV/c^2
        self.charge = charge
        self.is_antiparticle = is_antiparticle
        
    def __repr__(self):
        return f"{self.name}({'anti' if self.is_antiparticle else ''})"

class Deuteron(Particle):
    """Represents a deuteron particle."""
    
    def __init__(self, is_antideuteron: bool = False):
        super().__init__(
            name="deuteron",
            mass=1.87561294257,  # GeV/c^2
            charge=1,
            is_antiparticle=is_antideuteron
        )
        self.name = "antideuteron" if is_antideuteron else "deuteron"

class FireballState:
    """Represents the state of the quark-gluon fireball at a given time."""
    
    def __init__(self, temperature: float, density: float, time: float):
        self.temperature = temperature  # Kelvin
        self.density = density  # GeV/fm^3
        self.time = time  # seconds
    
    def __repr__(self):
        return f"Fireball(t={self.time:.2e}s, T={self.temperature:.2e}K, rho={self.density:.2e}GeV/fm³)"

class ParticleFormationModel:
    """Models the formation of fragile particles from the fireball."""
    
    # Energy thresholds for deuteron formation (GeV)
    DEUTERON_THRESHOLD = 0.5
    ANTI_DEUTERON_THRESHOLD = 0.5
    
    @staticmethod
    def calculate_formation_probability(temperature: float, density: float) -> float:
        """
        Calculate probability of deuteron formation based on temperature and density.
        
        Args:
            temperature: Current temperature in Kelvin
            density: Current density in GeV/fm^3
            
        Returns:
            Probability of formation (0-1)
        """
        # Convert temperature to GeV for consistency
        temp_gev = temperature * 8.617e-11  # Boltzmann constant in eV/K
        
        # Formation probability depends on temperature and density
        # Simplified model: higher density increases probability
        # Lower temperature decreases probability (more stable particles form)
        base_prob = min(1.0, density / 10.0) * max(0.0, 1.0 - temp_gev / 10.0)
        
        # Add some randomness to make it more realistic
        return min(1.0, max(0.0, base_prob + random.gauss(0, 0.1)))
    
    @staticmethod
    def simulate_decay_chain(initial_particles: List[Particle]) -> List[Particle]:
        """
        Simulate decay chain of particles to produce deuterons.
        
        Args:
            initial_particles: List of initial particles
            
        Returns:
            List of final particles including deuterons
        """
        # This is a simplified model - in reality this would be much more complex
        # For now, we'll just return the initial particles
        # In a real implementation, this would involve:
        # - Strong interaction processes
        # - Quantum chromodynamics calculations
        # - Statistical mechanics of particle interactions
        
        return initial_particles

class ParticleSimulator:
    """Main simulation class for modeling fragile matter formation."""
    
    def __init__(self, 
                 initial_temperature: float,
                 collision_energy: float,
                 duration: float,
                 time_step: float = 1e-25):
        """
        Initialize the particle simulator.
        
        Args:
            initial_temperature: Initial temperature in Kelvin
            collision_energy: Collision energy in GeV
            duration: Total simulation duration in seconds
            time_step: Time step for simulation in seconds
        """
        self.initial_temperature = initial_temperature
        self.collision_energy = collision_energy
        self.duration = duration
        self.time_step = time_step
        
        # Particle counts
        self.deuteron_count = 0
        self.antideuteron_count = 0
        
        # Track fireball states
        self.fireball_history: List[FireballState] = []
        
        # Formation model
        self.formation_model = ParticleFormationModel()
        
    def run_simulation(self) -> Dict:
        """
        Run the complete simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # Initialize fireball
        current_temp = self.initial_temperature
        current_density = 10.0  # Initial density in GeV/fm^3
        current_time = 0.0
        
        # Store initial state
        self.fireball_history.append(FireballState(current_temp, current_density, current_time))
        
        # Main simulation loop
        while current_time < self.duration:
            # Update temperature (cooling process)
            current_temp = self._calculate_cooling(current_time)
            
            # Update density (simplified)
            current_density = self._calculate_density(current_time)
            
            # Record state
            self.fireball_history.append(FireballState(current_temp, current_density, current_time))
            
            # Attempt to form particles
            self._attempt_particle_formation(current_temp, current_density)
            
            # Move to next time step
            current_time += self.time_step
            
        return {
            'deuteron_count': self.deuteron_count,
            'antideuteron_count': self.antideuteron_count,
            'fireball_history': self.fireball_history,
            'final_temperature': current_temp,
            'final_density': current_density
        }
    
    def _calculate_cooling(self, time: float) -> float:
        """
        Calculate temperature based on cooling curve.
        
        Args:
            time: Current time in seconds
            
        Returns:
            Temperature in Kelvin
        """
        # Simplified cooling model: exponential decay
        # In reality this would depend on the specific collision physics
        cooling_rate = 1e12  # K/s
        return max(1e6, self.initial_temperature * np.exp(-time * cooling_rate))
    
    def _calculate_density(self, time: float) -> float:
        """
        Calculate density based on time.
        
        Args:
            time: Current time in seconds
            
        Returns:
            Density in GeV/fm^3
        """
        # Simplified density evolution
        return max(0.1, 10.0 * np.exp(-time * 1e20))
    
    def _attempt_particle_formation(self, temperature: float, density: float):
        """
        Attempt to form particles at current conditions.
        
        Args:
            temperature: Current temperature in Kelvin
            density: Current density in GeV/fm^3
        """
        # Calculate formation probability
        prob = self.formation_model.calculate_formation_probability(temperature, density)
        
        # Generate random number for decision
        rand = random.random()
        
        # Form deuterons
        if rand < prob:
            # Check if we're above formation threshold
            if temperature > ParticleFormationModel.DEUTERON_THRESHOLD * 1e9:  # Convert to Kelvin
                self.deuteron_count += 1
                
        # Form antideuterons
        if rand < prob * 0.5:  # Lower probability for antideuterons
            if temperature > ParticleFormationModel.ANTI_DEUTERON_THRESHOLD * 1e9:
                self.antideuteron_count += 1

def visualize_results(results: Dict, save_plot: bool = False):
    """
    Visualize simulation results.
    
    Args:
        results: Dictionary from simulation run
        save_plot: Whether to save plot to file
    """
    try:
        import matplotlib.pyplot as plt
        
        # Extract data
        times = [state.time for state in results['fireball_history']]
        temps = [state.temperature for state in results['fireball_history']]
        densities = [state.density for state in results['fireball_history']]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Temperature plot
        ax1.plot(times, temps, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Fireball Temperature Evolution')
        ax1.grid(True)
        
        # Density plot
        ax2.plot(times, densities, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Density (GeV/fm³)')
        ax2.set_title('Fireball Density Evolution')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")