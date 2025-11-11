"""
Utility functions for the perovskite ML pipeline.
Includes seed setting, model save/load, and logging helpers.
"""

import os
import json
import joblib
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Global random seed for reproducibility
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for ML libraries
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"✓ Random seeds set to {seed}")


def setup_logging(log_dir: str = "logs", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_file: Name of log file (auto-generated if None)
    
    Returns:
        Logger object
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    if log_file is None:
        log_file = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_path = log_dir / log_file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None):
    """
    Save a trained model with optional metadata.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        metadata: Optional dictionary with model metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")
    
    if metadata:
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Metadata saved to {meta_path}")


def load_model(filepath: str, load_metadata: bool = True):
    """
    Load a trained model and optionally its metadata.
    
    Args:
        filepath: Path to the saved model
        load_metadata: Whether to load metadata file
    
    Returns:
        Model object, and metadata dict if load_metadata=True
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    
    if load_metadata:
        meta_path = filepath.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        else:
            print(f"⚠ Metadata file not found: {meta_path}")
            return model, None
    
    return model


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    print(f"✓ JSON saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Input file path
    
    Returns:
        Dictionary from JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def print_section(title: str, char: str = "=", width: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character for the border
        width: Total width of the header
    """
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width + "\n")


def count_features(X) -> int:
    """
    Count number of features in dataset.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
    
    Returns:
        Number of features
    """
    if hasattr(X, 'shape'):
        return X.shape[1]
    return 0


def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility.
    
    Returns:
        Dictionary with system details
    """
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'timestamp': get_timestamp(),
        'random_seed': RANDOM_SEED
    }
    
    # Get package versions
    packages = ['numpy', 'pandas', 'sklearn', 'lightgbm', 'xgboost', 
                'catboost', 'matminer', 'pymatgen', 'mp_api']
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            info[f'{pkg}_version'] = getattr(module, '__version__', 'unknown')
        except ImportError:
            info[f'{pkg}_version'] = 'not installed'
    
    return info


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # Test utility functions
    print_section("Testing Utility Functions")
    
    # Test seed setting
    set_seeds(42)
    
    # Test system info
    info = get_system_info()
    print("\nSystem Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Utility functions tested successfully")
