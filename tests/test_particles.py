import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from lhc_simulator.particles import Particle, Deuteron, Antideuteron, ParticleFactory

class TestParticle(unittest.TestCase):
    def test_particle_initialization(self):
        particle = Particle("Test", 1.0, 2.0)
        self.assertEqual(particle.name, "Test")
        self.assertEqual(particle.mass, 1.0)
        self.assertEqual(particle.charge, 2.0)

    def test_deuteron_initialization(self):
        deuteron = Deuteron()
        self.assertEqual(deuteron.name, "Deuteron")
        self.assertAlmostEqual(deuteron.mass, 1.8756129428e-27, places=15)
        self.assertEqual(deuteron.charge, 1.0)

    def test_antideuteron_initialization(self):
        antideuteron = Antideuteron()
        self.assertEqual(antideuteron.name, "Antideuteron")
        self.assertAlmostEqual(antideuteron.mass, 1.8756129428e-27, places=15)
        self.assertEqual(antideuteron.charge, -1.0)

class TestParticleFactory(unittest.TestCase):
    def test_create_deuteron(self):
        deuteron = ParticleFactory.create_particle("deuteron")
        self.assertIsInstance(deuteron, Deuteron)
        self.assertEqual(deuteron.name, "Deuteron")

    def test_create_antideuteron(self):
        antideuteron = ParticleFactory.create_particle("antideuteron")
        self.assertIsInstance(antideuteron, Antideuteron)
        self.assertEqual(antideuteron.name, "Antideuteron")

    def test_create_invalid_particle(self):
        with self.assertRaises(ValueError):
            ParticleFactory.create_particle("invalid")

if __name__ == '__main__':
    unittest.main()