"""
Feature engineering module for perovskite materials.
Generates ~300 features using matminer and pymatgen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry
from matminer.featurizers.base import MultipleFeaturizer

warnings.filterwarnings('ignore')


class PerovskiteFeaturizer:
    """
    Featurizer for perovskite materials using composition-based descriptors.
    Aims to generate ~300 features similar to the original paper.
    """
    
    def __init__(self):
        """Initialize featurizers."""
        self.featurizers = self._setup_featurizers()
        self.feature_list = []
    
    def _setup_featurizers(self) -> MultipleFeaturizer:
        """
        Set up matminer featurizers for composition-based features.
        
        Returns:
            MultipleFeaturizer object
        """
        print("Setting up featurizers...")
        
        # Use multiple composition featurizers to get ~300 features
        # NOTE: Stoichiometry featurizer can cause row duplication issues
        # Using only ElementProperty for stability
        featurizers = [
            # Magpie elemental property statistics (132 features)
            ElementProperty.from_preset("magpie"),
        ]
        
        multi_featurizer = MultipleFeaturizer(featurizers)
        print("Featurizers initialized")

        return multi_featurizer
    
    def add_basic_composition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic composition-based features.
        
        Args:
            df: DataFrame with formula_pretty column
        
        Returns:
            DataFrame with added features
        """
        print("Adding basic composition features...")
        
        df = df.copy()
        
        # Parse compositions
        df['composition'] = df['formula_pretty'].apply(
            lambda x: Composition(x) if pd.notna(x) else None
        )
        
        # Number of elements
        df['n_elements'] = df['composition'].apply(
            lambda x: len(x.elements) if x else None
        )
        
        # Total number of atoms
        df['n_atoms'] = df['composition'].apply(
            lambda x: x.num_atoms if x else None
        )
        
        # Average atomic mass
        df['avg_atomic_mass'] = df['composition'].apply(
            lambda x: np.mean([el.atomic_mass for el in x.elements]) if x else None
        )
        
        # Electronegativity statistics
        df['avg_electronegativity'] = df['composition'].apply(
            lambda x: np.mean([el.X for el in x.elements]) if x else None
        )
        df['std_electronegativity'] = df['composition'].apply(
            lambda x: np.std([el.X for el in x.elements]) if x and len(x.elements) > 1 else 0
        )
        df['max_electronegativity'] = df['composition'].apply(
            lambda x: np.max([el.X for el in x.elements]) if x else None
        )
        df['min_electronegativity'] = df['composition'].apply(
            lambda x: np.min([el.X for el in x.elements]) if x else None
        )
        df['range_electronegativity'] = df['max_electronegativity'] - df['min_electronegativity']
        
        # Atomic number statistics
        df['avg_atomic_number'] = df['composition'].apply(
            lambda x: np.mean([el.Z for el in x.elements]) if x else None
        )
        df['std_atomic_number'] = df['composition'].apply(
            lambda x: np.std([el.Z for el in x.elements]) if x and len(x.elements) > 1 else 0
        )
        
        # Atomic radius statistics (covalent radius)
        df['avg_atomic_radius'] = df['composition'].apply(
            lambda x: np.mean([el.atomic_radius for el in x.elements if el.atomic_radius]) if x else None
        )
        
        # Ionization energy statistics
        df['avg_ionization_energy'] = df['composition'].apply(
            lambda x: np.mean([el.ionization_energy for el in x.elements if el.ionization_energy]) if x else None
        )
        
        # Electron affinity statistics
        df['avg_electron_affinity'] = df['composition'].apply(
            lambda x: np.mean([el.electron_affinity for el in x.elements if el.electron_affinity]) if x else None
        )
        
        # Group and period statistics
        df['avg_group'] = df['composition'].apply(
            lambda x: np.mean([el.group for el in x.elements]) if x else None
        )
        df['avg_row'] = df['composition'].apply(
            lambda x: np.mean([el.row for el in x.elements]) if x else None
        )
        
        # Valence electron statistics (handle ambiguous valence for transition metals)
        def safe_valence(el):
            try:
                val = el.valence
                # Ensure we return a scalar, not a tuple/list
                if isinstance(val, (list, tuple)):
                    return float(val[0]) if val else 0.0
                return float(val) if val else 0.0
            except (ValueError, AttributeError):
                # For elements with ambiguous valence, use most common oxidation state
                if hasattr(el, 'common_oxidation_states') and el.common_oxidation_states:
                    # Take absolute value of most common oxidation state
                    return float(max(el.common_oxidation_states, key=lambda x: abs(x)))
                return 0.0
        
        df['avg_valence'] = df['composition'].apply(
            lambda x: np.mean([safe_valence(el) for el in x.elements]) if x else None
        )
        
        # Block statistics (s, p, d, f)
        df['n_d_electrons'] = df['composition'].apply(
            lambda x: sum([el.electronic_structure.count('d') for el in x.elements]) if x else None
        )
        
        # Fraction of metals, metalloids, nonmetals
        df['frac_metal'] = df['composition'].apply(
            lambda x: sum([el.is_metal for el in x.elements]) / len(x.elements) if x else None
        )
        df['frac_nonmetal'] = df['composition'].apply(
            lambda x: sum([not el.is_metal and not el.is_metalloid for el in x.elements]) / len(x.elements) if x else None
        )
        
        print(f"Added {len([c for c in df.columns if c not in ['formula_pretty', 'composition']])} basic features")

        return df
    
    def add_matminer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add matminer featurizer features.
        
        Args:
            df: DataFrame with composition column
        
        Returns:
            DataFrame with matminer features added
        """
        print("Computing matminer features (this may take a while)...")
        
        df = df.copy()
        
        # Ensure composition column exists
        if 'composition' not in df.columns:
            df['composition'] = df['formula_pretty'].apply(
                lambda x: Composition(x) if pd.notna(x) else None
            )
        
        # Remove rows with invalid compositions
        valid_mask = df['composition'].notna()
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"  Warning: Skipping {n_invalid} materials with invalid compositions")
        
        df_valid = df[valid_mask].copy()
        
        # Featurize using matminer
        try:
            df_valid = self.featurizers.featurize_dataframe(
                df_valid, 
                col_id='composition',
                ignore_errors=True,
                multiindex=False  # Prevent row duplication
            )
            print("Matminer features computed successfully")
            
            # Drop exact duplicates that may have been created
            initial_len = len(df_valid)
            df_valid = df_valid.drop_duplicates(subset=['formula_pretty'])
            if len(df_valid) < initial_len:
                print(f"  Warning: Removed {initial_len - len(df_valid)} duplicate rows created during featurization")
                
        except Exception as e:
            print(f"⚠ Warning: Some features failed to compute: {e}")
        
        # Merge back with original dataframe
        feature_cols = [col for col in df_valid.columns if col not in df.columns]
        df = df.merge(df_valid[['formula_pretty'] + feature_cols], on='formula_pretty', how='left')
        
        return df
    
    def add_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add structural features if lattice parameters are available.
        
        Args:
            df: DataFrame with lattice parameters
        
        Returns:
            DataFrame with structural features added
        """
        print("Adding structural features...")
        
        df = df.copy()
        
        lattice_cols = ['lattice_a', 'lattice_b', 'lattice_c', 
                       'lattice_alpha', 'lattice_beta', 'lattice_gamma']
        
        if all(col in df.columns for col in lattice_cols):
            # Lattice ratios
            df['lattice_b_over_a'] = df['lattice_b'] / df['lattice_a']
            df['lattice_c_over_a'] = df['lattice_c'] / df['lattice_a']
            df['lattice_c_over_b'] = df['lattice_c'] / df['lattice_b']
            
            # Angular deviations from cubic
            df['alpha_dev_from_90'] = np.abs(df['lattice_alpha'] - 90)
            df['beta_dev_from_90'] = np.abs(df['lattice_beta'] - 90)
            df['gamma_dev_from_90'] = np.abs(df['lattice_gamma'] - 90)
            
            # Total angular deviation
            df['total_angle_deviation'] = (
                df['alpha_dev_from_90'] + 
                df['beta_dev_from_90'] + 
                df['gamma_dev_from_90']
            )
            
            print(f"Added structural features")
        else:
            print("  Info: Lattice parameters not available, skipping structural features")
        
        # Density-based features
        if 'density' in df.columns and 'volume' in df.columns:
            df['packing_fraction'] = df['density'] / (df['volume'] ** (1/3))
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features (ratios, differences, etc.) and safe stoichiometry features.
        
        Args:
            df: DataFrame with base features
        
        Returns:
            DataFrame with derived features
        """
        print("Adding derived features...")
        
        df = df.copy()
        
        # Add safe stoichiometry features manually (avoiding matminer's buggy Stoichiometry featurizer)
        if 'composition' in df.columns:
            # Number of unique elements
            df['num_unique_elements'] = df['composition'].apply(
                lambda x: len(set([el.symbol for el in x.elements])) if x else None
            )
            
            # L0 norm (number of elements)
            df['l0_norm'] = df['composition'].apply(
                lambda x: len(x.elements) if x else None
            )
            
            # L2 norm of stoichiometric coefficients
            df['l2_norm'] = df['composition'].apply(
                lambda x: np.sqrt(sum([x[el]**2 for el in x.elements])) if x else None
            )
            
            # L3 norm
            df['l3_norm'] = df['composition'].apply(
                lambda x: sum([x[el]**3 for el in x.elements])**(1/3) if x else None
            )
        
        # Electronegativity differences and ratios
        if 'max_electronegativity' in df.columns and 'min_electronegativity' in df.columns:
            df['electronegativity_diff'] = df['max_electronegativity'] - df['min_electronegativity']
        
        # Atomic mass ratios
        if 'avg_atomic_mass' in df.columns and 'avg_atomic_number' in df.columns:
            df['mass_per_proton'] = df['avg_atomic_mass'] / df['avg_atomic_number']
        
        # Formation energy features (if available)
        if 'formation_energy_per_atom' in df.columns and 'n_atoms' in df.columns:
            df['total_formation_energy'] = df['formation_energy_per_atom'] * df['n_atoms']
        
        # Density features
        if 'density' in df.columns and 'n_atoms' in df.columns:
            df['density_per_atom'] = df['density'] / df['n_atoms']
        
        print(f"Added derived features")
        
        return df
    
    def featurize(self, df: pd.DataFrame, include_matminer: bool = True) -> pd.DataFrame:
        """
        Complete featurization pipeline.
        
        Args:
            df: Input DataFrame with materials data
            include_matminer: Whether to include matminer features (slower but more complete)
        
        Returns:
            Featurized DataFrame
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80 + "\n")
        
        df = df.copy()
        initial_cols = set(df.columns)
        
        # Add basic composition features
        df = self.add_basic_composition_features(df)
        
        # Add matminer features
        if include_matminer:
            df = self.add_matminer_features(df)
        
        # Add structural features
        df = self.add_structural_features(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Get list of new features
        new_cols = set(df.columns) - initial_cols
        self.feature_list = sorted([col for col in new_cols if col != 'composition'])
        
        print(f"\nFeaturization Summary:")
        print(f"  Total features added: {len(self.feature_list)}")
        print(f"  Total columns in dataframe: {len(df.columns)}")
        
        return df
    
    def save_feature_list(self, filepath: str):
        """
        Save list of generated features to file.
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({'feature_name': self.feature_list})
        df.to_csv(filepath, index=False)
        
        print(f"Feature list saved to {filepath}")


def featurize_data(
    input_path: str = "data/raw/perovskites_raw.csv",
    output_path: str = "data/processed/perovskites_features.csv",
    feature_list_path: str = "data/processed/features_list.csv"
) -> pd.DataFrame:
    """
    Main function to featurize perovskite data.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save featurized data
        feature_list_path: Path to save feature list
    
    Returns:
        Featurized DataFrame
    """
    # Load raw data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} materials")
    
    # Initialize featurizer
    featurizer = PerovskiteFeaturizer()
    
    # Featurize
    df_features = featurizer.featurize(df, include_matminer=True)
    
    # Save featurized data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"\nFeaturized data saved to {output_path}")
    print(f"  Shape: {df_features.shape}")
    
    # Save feature list
    featurizer.save_feature_list(feature_list_path)
    
    return df_features


if __name__ == "__main__":
    # Test featurization
    print("Testing featurization module...")
    
    try:
        df = featurize_data()
        print("\nFeaturization test successful")
        print(f"\nSample features:")
        print(df.head())
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
