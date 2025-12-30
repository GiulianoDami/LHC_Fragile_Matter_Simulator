import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from lhc_simulator.particles import Particle, Deuteron, Antideuteron, ParticleFactory

class TestParticle(unittest.TestCase):
    
    def test_particle_initialization(self):
        """Test basic particle initialization"""
        particle = Particle("test", 1.0, 2.0)
        self.assertEqual(particle.name, "test")
        self.assertEqual(particle.mass, 1.0)
        self.assertEqual(particle.charge, 2.0)
        
    def test_deuteron_initialization(self):
        """Test deuteron specific initialization"""
        deuteron = Deuteron()
        self.assertEqual(deuteron.name, "deuteron")
        self.assertEqual(deuteron.mass, 1.87561294257e-28)  # kg
        self.assertEqual(deuteron.charge, 1.0)
        
    def test_antideuteron_initialization(self):
        """Test antideuteron specific initialization"""
        antideuteron = Antideuteron()
        self.assertEqual(antideuteron.name, "antideuteron")
        self.assertEqual(antideuteron.mass, 1.87561294257e-28)  # kg
        self.assertEqual(antideuteron.charge, -1.0)
        
    def test_particle_factory_creation(self):
        """Test particle factory creates correct particle types"""
        # Test deuteron creation
        deuteron = ParticleFactory.create_particle("deuteron")
        self.assertIsInstance(deuteron, Deuteron)
        
        # Test antideuteron creation
        antideuteron = ParticleFactory.create_particle("antideuteron")
        self.assertIsInstance(antideuteron, Antideuteron)
        
        # Test unknown particle
        with self.assertRaises(ValueError):
            ParticleFactory.create_particle("unknown")

if __name__ == '__main__':
    unittest.main()