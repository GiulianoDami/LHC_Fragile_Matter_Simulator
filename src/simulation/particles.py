import numpy as np
from typing import Dict, List, Tuple
import random

class Particle:
    """Represents a fundamental particle in the simulation."""
    
    def __init__(self, mass: float, charge: int, name: str):
        self.mass = mass  # in GeV/c^2
        self.charge = charge
        self.name = name
        
    def __repr__(self):
        return f"{self.name}(mass={self.mass}, charge={self.charge})"

class Deuteron(Particle):
    """Deuteron particle - proton + neutron bound state."""
    
    def __init__(self):
        super().__init__(mass=1.875612942, charge=1, name="Deuteron")

class Antideuteron(Particle):
    """Antideuteron particle - antiproton + antineutron bound state."""
    
    def __init__(self):
        super().__init__(mass=1.875612942, charge=-1, name="Antideuteron")

class ParticleFactory:
    """Factory class for creating particles."""
    
    @staticmethod
    def create_particle(particle_type: str) -> Particle:
        """Create a particle instance based on type string."""
        if particle_type.lower() == "deuteron":
            return Deuteron()
        elif particle_type.lower() == "antideuteron":
            return Antideuteron()
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")

class ParticleFormationModel:
    """Models the formation of fragile particles through decay chains."""
    
    def __init__(self, temperature: float, density: float):
        self.temperature = temperature  # Kelvin
        self.density = density  # nucleons per fm^3
        
    def calculate_formation_probability(self, particle_type: str) -> float:
        """
        Calculate probability of forming a specific particle type.
        
        Args:
            particle_type: Type of particle to form
            
        Returns:
            Formation probability (0-1)
        """
        # Simplified model based on temperature and density
        base_temp = 1e12  # Reference temperature for deuteron formation
        temp_ratio = self.temperature / base_temp
        
        # Density factor (higher density increases formation probability)
        density_factor = min(1.0, self.density / 1.0)  # Normalize to typical density
        
        # Formation threshold (deuteron formation requires T < ~1e12 K)
        threshold = 1.0 if self.temperature < 1e12 else 0.0
        
        if particle_type.lower() == "deuteron":
            prob = temp_ratio * density_factor * threshold
            return max(0.0, min(1.0, prob))
        elif particle_type.lower() == "antideuteron":
            # Antideuteron formation is less likely due to annihilation
            prob = temp_ratio * density_factor * threshold * 0.1
            return max(0.0, min(1.0, prob))
        else:
            return 0.0
    
    def simulate_decay_chain(self, initial_particles: List[Particle]) -> List[Particle]:
        """
        Simulate decay chain from initial particles.
        
        Args:
            initial_particles: List of initial particles
            
        Returns:
            Final list of formed particles
        """
        formed_particles = []
        
        # Simple model: each initial particle has a chance to decay into deuterons
        for particle in initial_particles:
            # For simplicity, assume we're dealing with protons/neutrons
            # that can combine to form deuterons
            if random.random() < 0.3:  # 30% chance of deuteron formation
                formed_particles.append(ParticleFactory.create_particle("deuteron"))
                
        return formed_particles

class ParticleTracker:
    """Tracks particle counts and distributions throughout simulation."""
    
    def __init__(self):
        self.particle_counts: Dict[str, int] = {}
        self.history: List[Dict[str, int]] = []
        
    def update_counts(self, new_particles: List[Particle]):
        """Update particle counts with newly formed particles."""
        for particle in new_particles:
            count_key = particle.name
            self.particle_counts[count_key] = self.particle_counts.get(count_key, 0) + 1
            
    def record_state(self):
        """Record current particle state for history tracking."""
        self.history.append(self.particle_counts.copy())
        
    def get_total_count(self) -> int:
        """Get total number of all particles formed."""
        return sum(self.particle_counts.values())
        
    def get_counts(self) -> Dict[str, int]:
        """Get current particle counts."""
        return self.particle_counts.copy()