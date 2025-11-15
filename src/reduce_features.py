"""
Feature reduction module for perovskite materials.
Reduces features from ~293 to target counts: F10, F8
With 5,776 samples, max safe features ≈ 577 (rule: N/10)
But for robustness, targeting 8-10 features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import warnings
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


class FeatureReducer:
    """
    Reduces features through multiple filtering stages:
    1. Remove high missing (>10%), zero variance, high correlation (>0.92)
    2. LightGBM importance ranking
    3. RFE (Recursive Feature Elimination) for final subsets
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize feature reducer."""
        self.random_state = random_state
        self.feature_importance = None
        
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Load featurized data and separate features from target.
        
        Args:
            data_path: Path to featurized CSV
            
        Returns:
            X, y, feature_names
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        print(f"  Loaded {len(df)} materials with {len(df.columns)} columns")
        
        # Define columns to exclude from features
        exclude_cols = [
            'material_id', 'formula_pretty', 'composition', 'elements', 
            'structure', 'symmetry', 'band_gap', 'is_gap_direct',
            'deprecated', 'theoretical', 'crystal_system', 'spacegroup_symbol',
            'spacegroup_number'  # Keep this numeric one out too for now
        ]
        
        # Target
        target_col = 'band_gap'
        
        # Get feature columns (only numeric)
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col != target_col]
        
        # Remove rows with missing target
        df_clean = df[df[target_col].notna()].copy()
        print(f"  After removing missing targets: {len(df_clean)} materials")
        
        X = df_clean[feature_cols].copy()
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        y = df_clean[target_col].copy()
        
        print(f"  Features (numeric only): {len(X.columns)}")
        print(f"  Target range: {y.min():.3f} - {y.max():.3f} eV")
        
        return X, y, list(X.columns)
    
    def filter_by_missing(self, X: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        """
        Remove features with more than threshold missing values.
        
        Args:
            X: Feature matrix
            threshold: Maximum fraction of missing values (default 10%)
            
        Returns:
            Filtered feature matrix
        """
        print(f"\n[1] Filtering by missing values (threshold: {threshold*100:.0f}%)...")
        
        initial_features = X.shape[1]
        missing_pct = X.isnull().sum() / len(X)
        
        keep_features = missing_pct[missing_pct <= threshold].index.tolist()
        X_filtered = X[keep_features].copy()
        
        removed = initial_features - len(keep_features)
        print(f"  Removed {removed} features with >{threshold*100:.0f}% missing")
        print(f"  Remaining: {len(keep_features)} features")
        
        return X_filtered
    
    def filter_by_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with zero or near-zero variance.
        
        Args:
            X: Feature matrix
            
        Returns:
            Filtered feature matrix
        """
        print(f"\n[2] Filtering by variance...")
        
        initial_features = X.shape[1]
        
        # Impute missing values with median for variance calculation
        X_imputed = X.fillna(X.median())
        
        # Remove zero variance features
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(X_imputed)
        
        keep_features = X.columns[selector.get_support()].tolist()
        X_filtered = X[keep_features].copy()
        
        removed = initial_features - len(keep_features)
        print(f"  Removed {removed} features with zero variance")
        print(f"  Remaining: {len(keep_features)} features")
        
        return X_filtered
    
    def filter_by_correlation(self, X: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame:
        """
        Remove highly correlated features (keep one from each pair).
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold (default 0.92)
            
        Returns:
            Filtered feature matrix
        """
        print(f"\n[3] Filtering by correlation (threshold: {threshold})...")
        
        initial_features = X.shape[1]
        
        # Impute missing values with median for correlation calculation
        X_imputed = X.fillna(X.median())
        
        # Calculate correlation matrix
        corr_matrix = X_imputed.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        keep_features = [col for col in X.columns if col not in to_drop]
        X_filtered = X[keep_features].copy()
        
        removed = initial_features - len(keep_features)
        print(f"  Removed {removed} highly correlated features")
        print(f"  Remaining: {len(keep_features)} features")
        
        return X_filtered
    
    def rank_by_lightgbm(self, X: pd.DataFrame, y: pd.Series, top_n: int = 50) -> List[str]:
        """
        Rank features by LightGBM feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            top_n: Number of top features to return
            
        Returns:
            List of top feature names
        """
        print(f"\n[4] Ranking features by LightGBM importance (top {top_n})...")
        
        # Impute missing values
        X_imputed = X.fillna(X.median())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train LightGBM
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            verbose=-1
        )
        
        model.fit(X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance
        
        # Get top features
        top_features = importance.head(top_n)['feature'].tolist()
        
        print(f"  Top 10 most important features:")
        for i, row in importance.head(10).iterrows():
            print(f"    {row['feature']:40s}: {row['importance']:8.2f}")
        
        return top_features
    
    def select_by_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Select features using Recursive Feature Elimination (RFE).
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        print(f"\n[5] Selecting {n_features} features using RFE...")
        
        # Impute missing values
        X_imputed = X.fillna(X.median())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # RFE with LightGBM
        estimator = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1
        )
        
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X_scaled, y)
        
        selected_features = X.columns[selector.support_].tolist()
        
        print(f"  Selected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"    - {feat}")
        
        return selected_features
    
    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> float:
        """
        Evaluate a feature set using cross-validation.
        
        Args:
            X: Full feature matrix
            y: Target variable
            features: List of feature names to evaluate
            
        Returns:
            Mean CV score (R²)
        """
        X_subset = X[features].fillna(X[features].median())
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1
        )
        
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        return scores.mean()


def main():
    """
    Main feature reduction pipeline.
    Target: Create F10 and F8 feature subsets.
    """
    print("="*80)
    print("FEATURE REDUCTION PIPELINE")
    print("Target: 8-10 features for 5,776 samples")
    print("="*80)
    
    reducer = FeatureReducer(random_state=42)
    
    # Load data
    X, y, feature_names = reducer.load_data('data/processed/perovskites_features.csv')
    
    # Stage 1: Filter by missing values
    X = reducer.filter_by_missing(X, threshold=0.10)
    
    # Stage 2: Filter by variance
    X = reducer.filter_by_variance(X)
    
    # Stage 3: Filter by correlation
    X = reducer.filter_by_correlation(X, threshold=0.92)
    
    print(f"\nAfter initial filtering: {X.shape[1]} features remaining")
    
    # Stage 4: Rank by LightGBM importance (get top 30)
    top_30_features = reducer.rank_by_lightgbm(X, y, top_n=30)
    
    # Stage 5: Create feature subsets using RFE
    print("\n" + "="*80)
    print("CREATING FEATURE SUBSETS")
    print("="*80)
    
    # Use top 30 for RFE
    X_top30 = X[top_30_features]
    
    # Create multiple feature subsets to find optimal balance
    feature_configs = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 24, 26, 28, 30]
    feature_subsets = {}
    
    for n_feat in feature_configs:
        print(f"\n--- Feature Subset F{n_feat} ({n_feat} features) ---")
        features = reducer.select_by_rfe(X_top30, y, n_features=n_feat)
        score = reducer.evaluate_feature_set(X, y, features)
        print(f"  CV R² score: {score:.4f}")
        feature_subsets[f"F{n_feat}"] = {"features": features, "score": score}
    
    # Save feature subsets
    print("\n" + "="*80)
    print("SAVING FEATURE SUBSETS")
    print("="*80)
    
    output_dir = Path("results/feature_sets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each feature subset
    for subset_name, subset_data in feature_subsets.items():
        output_path = output_dir / f"feature_subset_{subset_name}.txt"
        with open(output_path, 'w') as f:
            n_features = len(subset_data['features'])
            f.write(f"# {subset_name} Feature Subset ({n_features} features)\n")
            f.write(f"# CV R² Score: {subset_data['score']:.4f}\n")
            f.write(f"# Created for 5,776 samples\n\n")
            for feat in subset_data['features']:
                f.write(f"{feat}\n")
        print(f"✓ Saved {subset_name} to: {output_path}")
    
    # Save feature importance rankings
    importance_path = output_dir / "feature_importance_rankings.csv"
    reducer.feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Saved importance rankings to: {importance_path}")
    
    # Find best performing subset
    best_subset = max(feature_subsets.items(), key=lambda x: x[1]['score'])
    best_name, best_data = best_subset
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE REDUCTION COMPLETE")
    print("="*80)
    print(f"\nOriginal features: {len(feature_names)}")
    print(f"After filtering: {X.shape[1]}")
    print(f"\nFeature Subsets Created:")
    for subset_name in sorted(feature_subsets.keys(), key=lambda x: int(x[1:])):
        score = feature_subsets[subset_name]['score']
        n_feat = len(feature_subsets[subset_name]['features'])
        best_marker = " ← BEST" if subset_name == best_name else ""
        print(f"  {subset_name}: {n_feat:2d} features (R² = {score:.4f}){best_marker}")
    
    print(f"\n✓ Best performing: {best_name} with R² = {best_data['score']:.4f}")
    print(f"\nFiles saved to: {output_dir}")
    
    return feature_subsets


if __name__ == "__main__":
    feature_subsets = main()
