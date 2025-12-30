import numpy as np

class RandomGenerator:
    """A utility class for generating random numbers and samples for the simulation."""
    
    def __init__(self, seed=None):
        """
        Initialize the random number generator.
        
        Args:
            seed (int, optional): Seed for the random number generator
        """
        if seed is not None:
            np.random.seed(seed)
    
    @staticmethod
    def generate_uniform(low=0.0, high=1.0, size=None):
        """
        Generate random numbers from a uniform distribution.
        
        Args:
            low (float): Lower bound of the distribution
            high (float): Upper bound of the distribution
            size (int or tuple): Output shape
            
        Returns:
            ndarray or scalar: Random numbers from the uniform distribution
        """
        return np.random.uniform(low, high, size)
    
    @staticmethod
    def generate_normal(mean=0.0, std=1.0, size=None):
        """
        Generate random numbers from a normal distribution.
        
        Args:
            mean (float): Mean of the distribution
            std (float): Standard deviation of the distribution
            size (int or tuple): Output shape
            
        Returns:
            ndarray or scalar: Random numbers from the normal distribution
        """
        return np.random.normal(mean, std, size)
    
    @staticmethod
    def generate_exponential(scale=1.0, size=None):
        """
        Generate random numbers from an exponential distribution.
        
        Args:
            scale (float): Scale parameter (1/lambda)
            size (int or tuple): Output shape
            
        Returns:
            ndarray or scalar: Random numbers from the exponential distribution
        """
        return np.random.exponential(scale, size)
    
    @staticmethod
    def generate_poisson(lam=1.0, size=None):
        """
        Generate random numbers from a Poisson distribution.
        
        Args:
            lam (float): Lambda parameter (expected value)
            size (int or tuple): Output shape
            
        Returns:
            ndarray or scalar: Random numbers from the Poisson distribution
        """
        return np.random.poisson(lam, size)
    
    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        """
        Generate a random sample from a given 1-D array.
        
        Args:
            a (1-D array-like): Input array
            size (int or tuple): Output shape
            replace (bool): Whether to sample with replacement
            p (1-D array-like): Probabilities for each element
            
        Returns:
            ndarray or scalar: Random sample(s) from the input array
        """
        return np.random.choice(a, size, replace, p)
    
    @staticmethod
    def shuffle(x):
        """
        Modify a sequence in-place by shuffling its contents.
        
        Args:
            x (array-like): The array to be shuffled
        """
        np.random.shuffle(x)
    
    @staticmethod
    def permutation(x):
        """
        Randomly permute a sequence, or return a permuted range.
        
        Args:
            x (array-like): The array to be shuffled
            
        Returns:
            ndarray: A new shuffled array
        """
        return np.random.permutation(x)