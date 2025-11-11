"""
Data ingestion and loading module for Materials Project perovskite data.
Handles downloading, loading, and initial processing of raw data.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mp_api.client import MPRester
from pymatgen.core import Structure, Composition
import warnings

warnings.filterwarnings('ignore')


def download_materials(query_config: Dict, out_json: str) -> str:
    """
    Download materials from Materials Project API.
    
    Args:
        query_config: Dictionary with query parameters
        out_json: Output JSON file path
    
    Returns:
        Path to saved JSON file
    """
    api_key = os.environ.get('MAPI_KEY') or os.environ.get('MP_API_KEY')
    
    if not api_key:
        raise ValueError("API key not found. Set MAPI_KEY or MP_API_KEY in environment")
    
    print(f"Connecting to Materials Project API...")
    
    with MPRester(api_key) as mpr:
        print(f"Searching for materials with parameters:")
        for key, value in query_config.items():
            if key != 'fields':
                print(f"  {key}: {value}")
        
        results = mpr.materials.summary.search(
            elements=query_config.get('elements'),
            num_elements=query_config.get('num_elements'),
            band_gap=query_config.get('band_gap'),
            energy_above_hull=query_config.get('energy_above_hull'),
            fields=query_config.get('fields', ['material_id', 'formula_pretty'])
        )
        
        print(f"Found {len(results)} materials")
        
        # Convert to dictionaries
        records = []
        for r in results:
            if hasattr(r, 'dict'):
                records.append(r.dict())
            else:
                records.append(r)
    
    # Save to JSON
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2, default=str)
    
    print(f"✓ Data saved to {out_path}")
    return str(out_path)


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw JSON data into pandas DataFrame.
    
    Args:
        path: Path to JSON file
    
    Returns:
        DataFrame with materials data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading data from {path}...")
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    print(f"✓ Loaded {len(df)} materials")
    return df


def load_existing_perovskite_data(data_dir: str = "materials_data") -> pd.DataFrame:
    """
    Load the existing perovskite data that was already downloaded.
    
    Args:
        data_dir: Directory containing materials data
    
    Returns:
        DataFrame with perovskite data
    """
    data_path = Path(data_dir)
    
    # Try to find the most complete dataset
    candidates = [
        data_path / "all_perovskite_complete_attributes.csv",
        data_path / "all_perovskite_materials_core.csv",
    ]
    
    for path in candidates:
        if path.exists():
            print(f"Loading existing perovskite data from {path}...")
            df = pd.read_csv(path)
            print(f"✓ Loaded {len(df)} perovskite materials")
            return df
    
    raise FileNotFoundError(f"No perovskite data found in {data_dir}")


def extract_structure_features(structure_dict: Dict) -> Dict:
    """
    Extract structural features from structure dictionary.
    
    Args:
        structure_dict: Structure data
    
    Returns:
        Dictionary with structural features
    """
    features = {}
    
    try:
        if isinstance(structure_dict, dict) and 'lattice' in structure_dict:
            lattice = structure_dict['lattice']
            
            # Extract lattice parameters
            if isinstance(lattice, dict):
                features['lattice_a'] = lattice.get('a')
                features['lattice_b'] = lattice.get('b')
                features['lattice_c'] = lattice.get('c')
                features['lattice_alpha'] = lattice.get('alpha')
                features['lattice_beta'] = lattice.get('beta')
                features['lattice_gamma'] = lattice.get('gamma')
                features['lattice_volume'] = lattice.get('volume')
            
            # Extract sites information
            if 'sites' in structure_dict:
                features['num_sites'] = len(structure_dict['sites'])
    except Exception as e:
        print(f"Warning: Could not extract structure features: {e}")
    
    return features


def process_to_raw_csv(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Process materials data and save as raw CSV with essential fields.
    
    Args:
        df: Input DataFrame with materials data
        output_path: Path to save processed CSV
    
    Returns:
        Processed DataFrame
    """
    print("Processing materials data...")
    
    # Essential columns to extract
    essential_cols = {
        'material_id': 'material_id',
        'formula_pretty': 'formula_pretty',
        'band_gap': 'band_gap',
        'is_gap_direct': 'is_gap_direct',
        'formation_energy_per_atom': 'formation_energy_per_atom',
        'energy_above_hull': 'energy_above_hull',
        'density': 'density',
        'volume': 'volume',
        'nsites': 'nsites',
        'nelements': 'nelements',
        'symmetry': 'symmetry',
    }
    
    # Create new dataframe with standardized columns
    processed = pd.DataFrame()
    
    for new_col, old_col in essential_cols.items():
        if old_col in df.columns:
            processed[new_col] = df[old_col]
        else:
            # Try alternative column names
            alt_cols = [col for col in df.columns if old_col.lower() in col.lower()]
            if alt_cols:
                processed[new_col] = df[alt_cols[0]]
            else:
                processed[new_col] = None
    
    # Extract symmetry information
    if 'symmetry' in df.columns:
        if isinstance(df['symmetry'].iloc[0], dict):
            processed['spacegroup_symbol'] = df['symmetry'].apply(
                lambda x: x.get('symbol') if isinstance(x, dict) else None
            )
            processed['spacegroup_number'] = df['symmetry'].apply(
                lambda x: x.get('number') if isinstance(x, dict) else None
            )
    
    # Extract elements from formula
    if 'formula_pretty' in processed.columns:
        processed['elements'] = processed['formula_pretty'].apply(
            lambda x: extract_elements_from_formula(x) if pd.notna(x) else None
        )
    
    # Extract lattice parameters if structure is available
    if 'structure' in df.columns:
        struct_features = df['structure'].apply(
            lambda x: extract_structure_features(x) if pd.notna(x) else {}
        )
        
        for feat in ['lattice_a', 'lattice_b', 'lattice_c', 
                     'lattice_alpha', 'lattice_beta', 'lattice_gamma', 'lattice_volume']:
            processed[feat] = struct_features.apply(lambda x: x.get(feat))
    
    # Remove rows with missing bandgap (critical for our task)
    initial_count = len(processed)
    processed = processed[processed['band_gap'].notna()]
    print(f"Removed {initial_count - len(processed)} materials with missing bandgap")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    
    print(f"✓ Processed data saved to {output_path}")
    print(f"  Shape: {processed.shape}")
    print(f"  Columns: {list(processed.columns)}")
    
    return processed


def extract_elements_from_formula(formula: str) -> List[str]:
    """
    Extract element symbols from chemical formula.
    
    Args:
        formula: Chemical formula string
    
    Returns:
        List of element symbols
    """
    try:
        comp = Composition(formula)
        return [str(el) for el in comp.elements]
    except:
        return []


def load_and_prepare_data(
    data_dir: str = "materials_data",
    output_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load existing perovskite data and prepare for processing.
    
    Args:
        data_dir: Directory with existing materials data
        output_dir: Directory to save processed data
    
    Returns:
        Tuple of (DataFrame, metadata dictionary)
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING PEROVSKITE DATA")
    print("="*80 + "\n")
    
    # Load existing data
    df = load_existing_perovskite_data(data_dir)
    
    # Process and save raw data
    raw_path = Path(output_dir) / "perovskites_raw.csv"
    processed_df = process_to_raw_csv(df, raw_path)
    
    # Create metadata
    metadata = {
        'source': data_dir,
        'n_materials': len(processed_df),
        'columns': list(processed_df.columns),
        'has_bandgap': processed_df['band_gap'].notna().sum(),
        'has_structure': 'lattice_a' in processed_df.columns,
        'bandgap_range': [
            float(processed_df['band_gap'].min()),
            float(processed_df['band_gap'].max())
        ] if 'band_gap' in processed_df.columns else None,
    }
    
    # Print summary
    print(f"\nData Summary:")
    print(f"  Total materials: {metadata['n_materials']}")
    print(f"  Materials with bandgap: {metadata['has_bandgap']}")
    print(f"  Bandgap range: {metadata['bandgap_range']}")
    print(f"  Has structural data: {metadata['has_structure']}")
    
    # Save metadata
    meta_path = Path(output_dir) / "data_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved to {meta_path}")
    
    return processed_df, metadata


if __name__ == "__main__":
    # Test data loading
    print("Testing data_io module...")
    
    try:
        df, metadata = load_and_prepare_data()
        print("\n✓ Data loading test successful")
        print(f"\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"\n✗ Error: {e}")
