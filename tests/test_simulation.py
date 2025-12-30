import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from lhc_simulator import ParticleSimulator

class TestParticleSimulator(unittest.TestCase):
    
    def setUp(self):
        self.simulator = ParticleSimulator(
            initial_temperature=1.5e12,
            collision_energy=13000,
            duration=1e-23
        )
    
    def test_initialization(self):
        self.assertEqual(self.simulator.initial_temperature, 1.5e12)
        self.assertEqual(self.simulator.collision_energy, 13000)
        self.assertEqual(self.simulator.duration, 1e-23)
        self.assertEqual(self.simulator.temperature, 1.5e12)
        self.assertEqual(self.simulator.deuteron_count, 0)
        self.assertEqual(self.simulator.antideuteron_count, 0)
    
    def test_temperature_evolution(self):
        # Test that temperature decreases over time
        initial_temp = self.simulator.temperature
        self.simulator._update_temperature(1e-24)
        self.assertLess(self.simulator.temperature, initial_temp)
    
    def test_particle_formation(self):
        # Mock the random number generator to control outcomes
        with patch('numpy.random.random', return_value=0.1):  # Below threshold
            self.simulator._form_particles()
            self.assertEqual(self.simulator.deuteron_count, 0)
            self.assertEqual(self.simulator.antideuteron_count, 0)
            
        with patch('numpy.random.random', return_value=0.9):  # Above threshold
            self.simulator._form_particles()
            self.assertIn(self.simulator.deuteron_count, [0, 1])
            self.assertIn(self.simulator.antideuteron_count, [0, 1])
    
    def test_run_simulation(self):
        results = self.simulator.run_simulation()
        
        self.assertIn('deuteron_count', results)
        self.assertIn('antideuteron_count', results)
        self.assertIn('temperature_history', results)
        self.assertIn('time_history', results)
        self.assertIsInstance(results['deuteron_count'], int)
        self.assertIsInstance(results['antideuteron_count'], int)
        self.assertIsInstance(results['temperature_history'], list)
        self.assertIsInstance(results['time_history'], list)
    
    def test_energy_threshold_calculation(self):
        threshold = self.simulator._calculate_energy_threshold()
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
    
    def test_deuteron_formation_probability(self):
        prob = self.simulator._calculate_deuteron_formation_probability(1e12)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_antideuteron_formation_probability(self):
        prob = self.simulator._calculate_antideuteron_formation_probability(1e12)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)

if __name__ == '__main__':
    unittest.main()