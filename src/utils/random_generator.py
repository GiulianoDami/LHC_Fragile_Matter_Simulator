import numpy as np
from typing import Tuple, List

class RandomGenerator:
    """
    A utility class for generating random numbers and samples
    used throughout the LHC Fragile Matter Simulator.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the random number generator with an optional seed.
        
        Args:
            seed (int): Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)
    
    @staticmethod
    def generate_uniform_sample(size: int = 1) -> np.ndarray:
        """
        Generate uniform random samples between 0 and 1.
        
        Args:
            size (int): Number of samples to generate
            
        Returns:
            np.ndarray: Array of uniform random samples
        """
        return np.random.uniform(0, 1, size)
    
    @staticmethod
    def generate_normal_sample(mean: float, std: float, size: int = 1) -> np.ndarray:
        """
        Generate normally distributed random samples.
        
        Args:
            mean (float): Mean of the normal distribution
            std (float): Standard deviation of the normal distribution
            size (int): Number of samples to generate
            
        Returns:
            np.ndarray: Array of normally distributed samples
        """
        return np.random.normal(mean, std, size)
    
    @staticmethod
    def generate_exponential_sample(scale: float, size: int = 1) -> np.ndarray:
        """
        Generate exponentially distributed random samples.
        
        Args:
            scale (float): Scale parameter (1/lambda) for exponential distribution
            size (int): Number of samples to generate
            
        Returns:
            np.ndarray: Array of exponentially distributed samples
        """
        return np.random.exponential(scale, size)
    
    @staticmethod
    def generate_poisson_sample(lam: float, size: int = 1) -> np.ndarray:
        """
        Generate Poisson-distributed random samples.
        
        Args:
            lam (float): Lambda parameter (mean) for Poisson distribution
            size (int): Number of samples to generate
            
        Returns:
            np.ndarray: Array of Poisson-distributed samples
        """
        return np.random.poisson(lam, size)
    
    @staticmethod
    def weighted_choice(choices: List[Tuple], weights: List[float]) -> Tuple:
        """
        Select an item from choices based on weighted probabilities.
        
        Args:
            choices (List[Tuple]): List of items to choose from
            weights (List[float]): Corresponding weights for each choice
            
        Returns:
            Tuple: Selected choice
        """
        total = sum(weights)
        r = np.random.uniform(0, total)
        upto = 0
        for choice, weight in zip(choices, weights):
            if upto + weight >= r:
                return choice
            upto += weight
        return choices[-1]  # Fallback
    
    @staticmethod
    def shuffle_array(arr: np.ndarray) -> np.ndarray:
        """
        Shuffle array elements randomly.
        
        Args:
            arr (np.ndarray): Array to shuffle
            
        Returns:
            np.ndarray: Shuffled array
        """
        shuffled = arr.copy()
        np.random.shuffle(shuffled)
        return shuffled
    
    @staticmethod
    def random_selection(arr: np.ndarray, k: int) -> np.ndarray:
        """
        Randomly select k elements from array without replacement.
        
        Args:
            arr (np.ndarray): Array to sample from
            k (int): Number of elements to select
            
        Returns:
            np.ndarray: Array of selected elements
        """
        return np.random.choice(arr, size=k, replace=False)