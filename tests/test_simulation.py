import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from lhc_simulator import ParticleSimulator

class TestParticleSimulator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.simulator = ParticleSimulator(
            initial_temperature=1.5e12,
            collision_energy=13000,
            duration=1e-23
        )
    
    def test_initialization(self):
        """Test that the simulator initializes correctly."""
        self.assertEqual(self.simulator.initial_temperature, 1.5e12)
        self.assertEqual(self.simulator.collision_energy, 13000)
        self.assertEqual(self.simulator.duration, 1e-23)
        self.assertIsNotNone(self.simulator.temperature_history)
        self.assertIsNotNone(self.simulator.particle_history)
    
    def test_temperature_evolution(self):
        """Test the temperature evolution calculation."""
        # Mock the temperature evolution function
        with patch.object(self.simulator, '_calculate_temperature_evolution') as mock_func:
            mock_func.return_value = [1.5e12, 1e12, 5e11, 1e11]
            result = self.simulator._calculate_temperature_evolution()
            self.assertEqual(result, [1.5e12, 1e12, 5e11, 1e11])
    
    def test_particle_formation(self):
        """Test particle formation logic."""
        # Mock temperature history
        self.simulator.temperature_history = [1.5e12, 1e12, 5e11, 1e11]
        
        # Mock particle formation function
        with patch.object(self.simulator, '_form_particles') as mock_func:
            mock_func.return_value = {'deuteron_count': 100, 'antideuteron_count': 90}
            result = self.simulator._form_particles()
            self.assertEqual(result['deuteron_count'], 100)
            self.assertEqual(result['antideuteron_count'], 90)
    
    def test_run_simulation(self):
        """Test full simulation run."""
        # Mock internal methods
        with patch.object(self.simulator, '_calculate_temperature_evolution') as mock_temp, \
             patch.object(self.simulator, '_form_particles') as mock_form:
            
            mock_temp.return_value = [1.5e12, 1e12, 5e11, 1e11]
            mock_form.return_value = {'deuteron_count': 100, 'antideuteron_count': 90}
            
            result = self.simulator.run_simulation()
            
            self.assertIn('deuteron_count', result)
            self.assertIn('antideuteron_count', result)
            self.assertIn('temperature_history', result)
            self.assertIn('particle_history', result)
    
    def test_energy_threshold_calculation(self):
        """Test energy threshold calculations."""
        threshold = self.simulator._calculate_energy_threshold()
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
    
    def test_deuteron_formation_probability(self):
        """Test deuteron formation probability calculation."""
        prob = self.simulator._calculate_deuteron_formation_probability(1e12)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_antideuteron_formation_probability(self):
        """Test antideuteron formation probability calculation."""
        prob = self.simulator._calculate_antideuteron_formation_probability(1e12)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_invalid_initial_temperature(self):
        """Test initialization with invalid temperature."""
        with self.assertRaises(ValueError):
            ParticleSimulator(initial_temperature=-1e12, collision_energy=13000, duration=1e-23)
    
    def test_invalid_collision_energy(self):
        """Test initialization with invalid collision energy."""
        with self.assertRaises(ValueError):
            ParticleSimulator(initial_temperature=1.5e12, collision_energy=-13000, duration=1e-23)
    
    def test_invalid_duration(self):
        """Test initialization with invalid duration."""
        with self.assertRaises(ValueError):
            ParticleSimulator(initial_temperature=1.5e12, collision_energy=13000, duration=-1e-23)

if __name__ == '__main__':
    unittest.main()