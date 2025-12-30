import json
import pickle
from typing import Dict, Any, List
import numpy as np


class DataHandler:
    """Handles data input/output operations for the LHC Fragile Matter Simulator."""
    
    @staticmethod
    def save_simulation_results(results: Dict[str, Any], filename: str) -> None:
        """
        Save simulation results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Simulation results dictionary
            filename (str): Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving results: {e}")
    
    @staticmethod
    def load_simulation_results(filename: str) -> Dict[str, Any]:
        """
        Load simulation results from a JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            Dict[str, Any]: Loaded simulation results
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    @staticmethod
    def save_numpy_array(data: np.ndarray, filename: str) -> None:
        """
        Save numpy array to binary file.
        
        Args:
            data (np.ndarray): Numpy array to save
            filename (str): Output filename
        """
        try:
            np.save(filename, data)
        except Exception as e:
            print(f"Error saving numpy array: {e}")
    
    @staticmethod
    def load_numpy_array(filename: str) -> np.ndarray:
        """
        Load numpy array from binary file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            np.ndarray: Loaded numpy array
        """
        try:
            return np.load(filename)
        except Exception as e:
            print(f"Error loading numpy array: {e}")
            return np.array([])
    
    @staticmethod
    def save_pickle_data(data: Any, filename: str) -> None:
        """
        Save arbitrary Python object using pickle.
        
        Args:
            data (Any): Object to save
            filename (str): Output filename
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving pickle data: {e}")
    
    @staticmethod
    def load_pickle_data(filename: str) -> Any:
        """
        Load Python object from pickle file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            Any: Loaded Python object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle data: {e}")
            return None
    
    @staticmethod
    def save_particle_history(history: List[Dict[str, Any]], filename: str) -> None:
        """
        Save particle formation history to JSON file.
        
        Args:
            history (List[Dict[str, Any]]): List of particle state dictionaries
            filename (str): Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving particle history: {e}")
    
    @staticmethod
    def load_particle_history(filename: str) -> List[Dict[str, Any]]:
        """
        Load particle formation history from JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            List[Dict[str, Any]]: Loaded particle history
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading particle history: {e}")
            return []