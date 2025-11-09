"""
Configuration module for Materials Project API downloader.
Loads settings from environment variables (.env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class that loads settings from environment variables."""
    
    # API Keys
    MP_API_KEY = os.getenv('MP_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Application Settings
    MAX_RESULTS_PER_QUERY = int(os.getenv('MAX_RESULTS_PER_QUERY', 1000))
    OUTPUT_DIRECTORY = Path(os.getenv('OUTPUT_DIRECTORY', 'materials_data'))
    DEFAULT_ENERGY_ABOVE_HULL_MAX = float(os.getenv('DEFAULT_ENERGY_ABOVE_HULL_MAX', 0.1))
    
    # Default search parameters
    DEFAULT_BAND_GAP_RANGE = (0.0, 10.0)
    DEFAULT_NSITES_RANGE = (1, 200)  # Keep original name for search compatibility
    
    # Common element groups for searching
    TRANSITION_METALS = [
        "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    ]
    
    LANTHANIDES = [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", 
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
    ]
    
    SEMICONDUCTORS = ["Si", "Ge", "Ga", "As", "In", "P", "Sb", "Te"]
    
    OXIDES_COMMON = ["Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Al", "Ti", "Fe", "O"]
    
    @classmethod
    def validate_api_key(cls):
        """Check if the Materials Project API key is properly configured."""
        if not cls.MP_API_KEY or cls.MP_API_KEY == 'your_materials_project_api_key_here':
            return False, "Materials Project API key not configured"
        return True, "API key is configured"
    
    @classmethod
    def get_api_key(cls):
        """Get the Materials Project API key, with validation."""
        is_valid, message = cls.validate_api_key()
        if not is_valid:
            raise ValueError(f"API Key Error: {message}. Please set MP_API_KEY in your .env file.")
        return cls.MP_API_KEY
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without revealing API keys)."""
        print("Current Configuration:")
        print("=" * 40)
        print(f"MP API Key configured: {'Yes' if cls.MP_API_KEY else 'No'}")
        print(f"Max results per query: {cls.MAX_RESULTS_PER_QUERY}")
        print(f"Output directory: {cls.OUTPUT_DIRECTORY}")
        print(f"Default energy above hull max: {cls.DEFAULT_ENERGY_ABOVE_HULL_MAX} eV/atom")
        print(f"Default band gap range: {cls.DEFAULT_BAND_GAP_RANGE} eV")
        print(f"Default nsites range: {cls.DEFAULT_NSITES_RANGE}")


# Create a default instance for easy importing
config = Config()