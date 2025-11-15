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


def load_double_perovskite_data(
    raw_path: str = "data/raw/double_perovskites_raw.csv",
    features_path: str = "data/processed/perovskites_features.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load double perovskite raw and featurized data.
    
    Args:
        raw_path: Path to raw double perovskite CSV
        features_path: Path to featurized data CSV
    
    Returns:
        Tuple of (raw_df, features_df)
    """
    raw_path = Path(raw_path)
    features_path = Path(features_path)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Featurized data not found: {features_path}")
    
    print(f"Loading double perovskite data...")
    print(f"  Raw: {raw_path}")
    print(f"  Features: {features_path}")
    
    raw_df = pd.read_csv(raw_path)
    features_df = pd.read_csv(features_path)
    
    print(f"✓ Raw data: {raw_df.shape[0]} materials, {raw_df.shape[1]} columns")
    print(f"✓ Feature data: {features_df.shape[0]} materials, {features_df.shape[1]} columns")
    
    return raw_df, features_df


def load_feature_subset(subset_name: str, features_dir: str = "results/feature_sets") -> List[str]:
    """
    Load a specific feature subset (e.g., F10, F22).
    
    Args:
        subset_name: Name of subset (e.g., 'F10', 'F22')
        features_dir: Directory containing feature subset files
    
    Returns:
        List of feature names
    """
    features_dir = Path(features_dir)
    subset_path = features_dir / f"feature_subset_{subset_name}.txt"
    
    if not subset_path.exists():
        raise FileNotFoundError(f"Feature subset not found: {subset_path}")
    
    print(f"Loading feature subset {subset_name} from {subset_path}...")
    
    features = []
    with open(subset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('CV') and not line.startswith('Selected'):
                # Extract feature name (remove leading "- " if present)
                if line.startswith('- '):
                    line = line[2:]
                if line:
                    features.append(line)
    
    print(f"✓ Loaded {len(features)} features for {subset_name}")
    return features


def prepare_training_data(
    subset_name: str = "F10",
    features_path: str = "data/processed/perovskites_features.csv",
    features_dir: str = "results/feature_sets",
    task: str = 'regression'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare training data with specific feature subset.
    
    Args:
        subset_name: Feature subset to use (e.g., 'F10', 'F22')
        features_path: Path to full featurized data
        features_dir: Directory with feature subset definitions
        task: 'regression' (bandgap) or 'classification' (gap type)
    
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature DataFrame
        - y: Target Series (band_gap for regression, is_gap_direct for classification)
        - feature_names: List of feature names
    """
    print("\n" + "="*80)
    print(f"PREPARING TRAINING DATA - Feature Subset: {subset_name}, Task: {task.upper()}")
    print("="*80 + "\n")
    
    # Load full featurized data
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Featurized data not found: {features_path}")
    
    df = pd.read_csv(features_path)
    print(f"Loaded full dataset: {df.shape}")
    
    # Load specific feature subset
    feature_names = load_feature_subset(subset_name, features_dir)
    
    # Verify all features exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    # Extract features and target
    X = df[feature_names].copy()
    
    # Select target based on task
    if task == 'regression':
        target_col = 'band_gap'
        y = df[target_col].copy()
    else:  # classification
        target_col = 'is_gap_direct'
        if target_col not in df.columns:
            raise ValueError(f"Classification target '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")
        y = df[target_col].copy()
    
    # Remove rows with missing target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    
    if task == 'regression':
        print(f"  Target (band_gap) range: {y.min():.3f} - {y.max():.3f} eV")
    else:  # classification
        print(f"  Target (is_gap_direct) distribution:")
        value_counts = y.value_counts()
        for val, count in value_counts.items():
            label = "Direct" if val else "Indirect"
            pct = count / len(y) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")
    
    print(f"  Missing values in X: {X.isnull().sum().sum()}")
    
    return X, y, feature_names


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
        # Test loading double perovskite data
        print("\n1. Testing double perovskite data loading...")
        raw_df, features_df = load_double_perovskite_data()
        print(f"✓ Successfully loaded {len(raw_df)} materials")
        
        # Test loading feature subsets
        print("\n2. Testing feature subset loading...")
        f10_features = load_feature_subset("F10")
        print(f"✓ F10 features: {f10_features}")
        
        f22_features = load_feature_subset("F22")
        print(f"✓ F22 features: {f22_features}")
        
        # Test preparing training data
        print("\n3. Testing training data preparation...")
        X_f10, y_f10, features_f10 = prepare_training_data("F10")
        print(f"✓ F10 data prepared: X shape = {X_f10.shape}, y shape = {y_f10.shape}")
        
        X_f22, y_f22, features_f22 = prepare_training_data("F22")
        print(f"✓ F22 data prepared: X shape = {X_f22.shape}, y shape = {y_f22.shape}")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

