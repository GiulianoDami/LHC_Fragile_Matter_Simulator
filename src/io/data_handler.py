import json
import pickle
from typing import Dict, Any, List
import numpy as np


class DataHandler:
    """Handles input/output operations for simulation data."""
    
    @staticmethod
    def save_simulation_data(data: Dict[str, Any], filename: str) -> None:
        """
        Save simulation results to a JSON file.
        
        Args:
            data: Dictionary containing simulation results
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            raise IOError(f"Failed to save data to {filename}: {str(e)}")
    
    @staticmethod
    def load_simulation_data(filename: str) -> Dict[str, Any]:
        """
        Load simulation results from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary containing loaded simulation results
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load data from {filename}: {str(e)}")
    
    @staticmethod
    def save_numpy_array(array: np.ndarray, filename: str) -> None:
        """
        Save numpy array to binary file.
        
        Args:
            array: Numpy array to save
            filename: Output filename
        """
        try:
            np.save(filename, array)
        except Exception as e:
            raise IOError(f"Failed to save numpy array to {filename}: {str(e)}")
    
    @staticmethod
    def load_numpy_array(filename: str) -> np.ndarray:
        """
        Load numpy array from binary file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded numpy array
        """
        try:
            return np.load(filename)
        except Exception as e:
            raise IOError(f"Failed to load numpy array from {filename}: {str(e)}")
    
    @staticmethod
    def save_pickle_data(data: Any, filename: str) -> None:
        """
        Save arbitrary Python object using pickle.
        
        Args:
            data: Object to save
            filename: Output filename
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise IOError(f"Failed to save pickle data to {filename}: {str(e)}")
    
    @staticmethod
    def load_pickle_data(filename: str) -> Any:
        """
        Load Python object from pickle file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded Python object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load pickle data from {filename}: {str(e)}")
    
    @staticmethod
    def save_particle_history(history: List[Dict[str, Any]], filename: str) -> None:
        """
        Save particle formation history to file.
        
        Args:
            history: List of particle state dictionaries
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save particle history to {filename}: {str(e)}")
    
    @staticmethod
    def load_particle_history(filename: str) -> List[Dict[str, Any]]:
        """
        Load particle formation history from file.
        
        Args:
            filename: Input filename
            
        Returns:
            List of particle state dictionaries
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load particle history from {filename}: {str(e)}")