"""
Preprocessing module for feature preparation, imputation, scaling, and splitting.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles all preprocessing steps for ML pipeline.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.target_col = 'band_gap'
        self.classification_target_col = 'is_gap_direct'
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate materials based on formula and spacegroup.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame without duplicates
        """
        print("Removing duplicates...")
        initial_count = len(df)
        
        # Remove duplicates based on formula_pretty and spacegroup_symbol
        if 'formula_pretty' in df.columns and 'spacegroup_symbol' in df.columns:
            df = df.drop_duplicates(subset=['formula_pretty', 'spacegroup_symbol'])
        else:
            df = df.drop_duplicates(subset=['formula_pretty'])
        
        removed = initial_count - len(df)
        print(f"✓ Removed {removed} duplicates ({removed/initial_count*100:.1f}%)")
        
        return df
    
    def separate_features_target(
        self, 
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features from target variable.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            exclude_cols: Columns to exclude from features
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_col is None:
            target_col = self.target_col
        
        if exclude_cols is None:
            exclude_cols = [
                'material_id', 'formula_pretty', 'composition', 
                'elements', 'structure', 'symmetry',
                target_col
            ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Separate X and y
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        self.feature_names = feature_cols
        
        print(f"Features: {len(feature_cols)} columns")
        print(f"Target: {target_col}")
        print(f"Samples: {len(df)}")
        
        return X, y
    
    def impute_missing_values(
        self, 
        X: pd.DataFrame, 
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Impute missing values using specified strategy.
        
        Args:
            X: Feature DataFrame
            strategy: Imputation strategy ('mean', 'median', 'knn', 'mice', 'zero')
        
        Returns:
            Imputed DataFrame
        """
        print(f"Imputing missing values using '{strategy}' strategy...")
        
        # Check missing values
        missing_count = X.isnull().sum().sum()
        missing_pct = missing_count / (X.shape[0] * X.shape[1]) * 100
        print(f"  Missing values: {missing_count} ({missing_pct:.2f}%)")
        
        if missing_count == 0:
            print("  No missing values found, skipping imputation")
            return X
        
        X_imputed = X.copy()

        # Coerce to numeric where possible (booleans->ints, numeric-like strings->numbers).
        # Non-convertible values become NaN so that imputer can handle them.
        X_imputed = X_imputed.apply(pd.to_numeric, errors='coerce')

        # Detect columns that are entirely NaN (some featurizers add empty structural columns).
        all_nan_cols = X_imputed.columns[X_imputed.isnull().all()].tolist()
        if all_nan_cols:
            print(f"  Found {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
            print(f"  Dropping these columns as they contain no information...")
            X_imputed = X_imputed.drop(columns=all_nan_cols)
            # Update feature names
            if self.feature_names:
                self.feature_names = [f for f in self.feature_names if f not in all_nan_cols]

        if strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif strategy == 'zero':
            self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        elif strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        elif strategy == 'mice':
            self.imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        # Fit and transform
        X_imputed_array = self.imputer.fit_transform(X_imputed)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X_imputed.columns, index=X_imputed.index)
        
        print(f"✓ Imputation complete")
        
        return X_imputed
    
    def scale_features(
        self, 
        X: pd.DataFrame, 
        scaler_type: str = 'robust'
    ) -> pd.DataFrame:
        """
        Scale features using specified scaler.
        
        Args:
            X: Feature DataFrame
            scaler_type: Type of scaler ('robust' or 'standard')
        
        Returns:
            Scaled DataFrame
        """
        print(f"Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_scaled_array = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
        
        print(f"✓ Scaling complete")
        
        return X_scaled
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Fraction of data for test set
            stratify: Whether to stratify split (for classification)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data (test_size={test_size})...")
        
        stratify_col = None
        if stratify:
            # Create bins for stratification
            y_binned = pd.cut(y, bins=5, labels=False)
            stratify_col = y_binned
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sampling_strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for handling class imbalance (classification only).
        
        Args:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: SMOTE sampling strategy
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        print("Applying SMOTE for class balancing...")
        
        # Check class distribution
        class_counts = y_train.value_counts()
        print(f"  Original class distribution:\n{class_counts}")
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        class_counts_new = y_resampled.value_counts()
        print(f"  New class distribution:\n{class_counts_new}")
        print(f"✓ SMOTE applied: {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def save_preprocessor(self, output_dir: str, prefix: str = ''):
        """
        Save scaler and imputer for later use.
        
        Args:
            output_dir: Directory to save preprocessor objects
            prefix: Prefix for saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.scaler:
            scaler_path = output_dir / f"{prefix}scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Scaler saved to {scaler_path}")
        
        if self.imputer:
            imputer_path = output_dir / f"{prefix}imputer.pkl"
            joblib.dump(self.imputer, imputer_path)
            print(f"✓ Imputer saved to {imputer_path}")
        
        if self.feature_names:
            features_path = output_dir / f"{prefix}feature_names.txt"
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_names))
            print(f"✓ Feature names saved to {features_path}")


def preprocess_data(
    input_path: str = "data/processed/perovskites_features.csv",
    output_dir: str = "data/processed",
    imputation_strategy: str = 'mean',
    test_size: float = 0.2,
    remove_metals: bool = False,
    for_classification: bool = False
) -> Dict:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_path: Path to featurized data
        output_dir: Directory to save processed data
        imputation_strategy: Strategy for missing value imputation
        test_size: Fraction for test set
        remove_metals: Whether to exclude metallic materials
        for_classification: Whether to prepare for classification task
    
    Returns:
        Dictionary with processed data and metadata
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} materials")
    
    # Remove metals if requested
    if remove_metals:
        print("\nRemoving metallic materials (band_gap < 0.1 eV)...")
        initial_count = len(df)
        df = df[df['band_gap'] >= 0.1]
        removed = initial_count - len(df)
        print(f"Removed {removed} metallic materials")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=42)
    
    # Remove duplicates
    df = preprocessor.remove_duplicates(df)
    
    # Separate features and target
    target_col = 'is_gap_direct' if for_classification else 'band_gap'
    X, y = preprocessor.separate_features_target(df, target_col=target_col)
    
    # Impute missing values
    X = preprocessor.impute_missing_values(X, strategy=imputation_strategy)
    
    # Scale features
    X = preprocessor.scale_features(X, scaler_type='robust')
    
    # Split data
    stratify = for_classification
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y, test_size=test_size, stratify=stratify
    )
    
    # Apply SMOTE for classification if needed
    if for_classification:
        # Check if classes are imbalanced
        class_counts = y_train.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio > 1.5:
            X_train, y_train = preprocessor.apply_smote(X_train, y_train)
    
    # Save preprocessor objects
    prefix = f"{imputation_strategy}_{'classification' if for_classification else 'regression'}_"
    if remove_metals:
        prefix += "nonmetals_"
    
    preprocessor.save_preprocessor(output_dir, prefix=prefix)
    
    # Save processed data
    output_dir = Path(output_dir)
    
    # Save split indices
    split_indices = {
        'train_idx': X_train.index.tolist(),
        'test_idx': X_test.index.tolist()
    }
    split_path = output_dir / f"{prefix}split_indices.pkl"
    joblib.dump(split_indices, split_path)
    print(f"✓ Split indices saved to {split_path}")
    
    # Save processed datasets
    joblib.dump(X_train, output_dir / f"{prefix}X_train.pkl")
    joblib.dump(X_test, output_dir / f"{prefix}X_test.pkl")
    joblib.dump(y_train, output_dir / f"{prefix}y_train.pkl")
    joblib.dump(y_test, output_dir / f"{prefix}y_test.pkl")
    print(f"✓ Processed datasets saved to {output_dir}")
    
    # Create metadata
    metadata = {
        'n_samples': len(df),
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'imputation_strategy': imputation_strategy,
        'test_size': test_size,
        'remove_metals': remove_metals,
        'task': 'classification' if for_classification else 'regression',
        'target_col': target_col,
        'feature_names': preprocessor.feature_names
    }
    
    # Save metadata
    import json
    meta_path = output_dir / f"{prefix}preprocessing_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved to {meta_path}")
    
    print("\n✓ Preprocessing complete!")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': metadata,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Test preprocessing with multiple strategies
    print("Testing preprocessing module...")
    
    strategies = ['mean', 'knn']
    
    for strategy in strategies:
        try:
            print(f"\n{'='*80}")
            print(f"Testing with {strategy} imputation")
            print(f"{'='*80}")
            
            result = preprocess_data(
                imputation_strategy=strategy,
                remove_metals=False,
                for_classification=False
            )
            
            print(f"\n✓ Preprocessing test successful for {strategy}")
            print(f"  X_train shape: {result['X_train'].shape}")
            print(f"  X_test shape: {result['X_test'].shape}")
            
        except Exception as e:
            print(f"\n✗ Error with {strategy}: {e}")
            import traceback
            traceback.print_exc()
