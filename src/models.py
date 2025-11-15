"""
Model training module with multiple ML algorithms.
Includes LightGBM, XGBoost, CatBoost, Random Forest, and MLP.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Wrapper for training and evaluating various ML models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
    
    def get_default_params(self, model_name: str, task: str = 'regression') -> Dict:
        """
        Get default hyperparameters for each model.
        
        Args:
            model_name: Name of the model
            task: 'regression' or 'classification'
        
        Returns:
            Dictionary of default parameters
        """
        if task == 'regression':
            params = {
                'lgbm': {
                    'n_estimators': 2000,  # Increased for better convergence
                    'learning_rate': 0.05,  # Slightly higher for small dataset
                    'num_leaves': 31,  # Reduced to prevent overfitting on small data
                    'max_depth': 8,  # Limited depth for regularization
                    'min_child_samples': 10,  # Prevent overfitting
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 1.0,  # L2 regularization
                    'random_state': self.random_state,
                    'verbose': -1,
                    'force_col_wise': True  # Better for many features
                },
                'xgb': {
                    'n_estimators': 2000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 3,  # Prevent overfitting
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': self.random_state,
                    'base_score': 0.5  # Fix for SHAP: must be numeric, not string
                },
                'catboost': {
                    'iterations': 2000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,  # Stronger regularization
                    'random_state': self.random_state,
                    'verbose': False
                },
                'rf': {
                    'n_estimators': 500,  # More trees
                    'max_depth': 15,  # Reduced depth
                    'min_samples_split': 10,  # Increased for regularization
                    'min_samples_leaf': 4,  # Increased for regularization
                    'max_features': 'sqrt',  # Feature subsampling
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'mlp': {
                    'hidden_layer_sizes': (128, 64, 32),  # Deeper network
                    'activation': 'relu',
                    'alpha': 0.01,  # L2 regularization
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': self.random_state
                }
            }
        else:  # classification
            params = {
                'lgbm': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'max_depth': 8,
                    'min_child_samples': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': self.random_state,
                    'verbose': -1,
                    'force_col_wise': True
                },
                'xgb': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 2,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': self.random_state,
                    'base_score': 0.5  # Fix for SHAP: must be numeric, not string
                },
                'catboost': {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,
                    'random_state': self.random_state,
                    'verbose': False
                },
                'rf': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'max_features': 'sqrt',
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'mlp': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'alpha': 0.01,
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': self.random_state
                }
            }
        
        return params.get(model_name, {})
    
    def get_model(self, model_name: str, task: str = 'regression', params: Optional[Dict] = None):
        """
        Get model instance with specified parameters.
        
        Args:
            model_name: Name of the model
            task: 'regression' or 'classification'
            params: Model parameters (uses defaults if None)
        
        Returns:
            Model instance
        """
        if params is None:
            params = self.get_default_params(model_name, task)
        
        if task == 'regression':
            models_map = {
                'lgbm': lgb.LGBMRegressor,
                'xgb': xgb.XGBRegressor,
                'catboost': CatBoostRegressor,
                'rf': RandomForestRegressor,
                'svr': SVR,
                'mlp': MLPRegressor
            }
        else:
            models_map = {
                'lgbm': lgb.LGBMClassifier,
                'xgb': xgb.XGBClassifier,
                'catboost': CatBoostClassifier,
                'rf': RandomForestClassifier,
                'logistic': LogisticRegression,
                'mlp': MLPClassifier
            }
        
        model_class = models_map.get(model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_class(**params)
    
    def train_lgbm_regression(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None, 
        params: Optional[Dict] = None
    ):
        """
        Train LightGBM regressor (primary model).
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model parameters
        
        Returns:
            Trained model
        """
        print("Training LightGBM Regressor...")
        
        if params is None:
            params = self.get_default_params('lgbm', 'regression')
        
        model = lgb.LGBMRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        else:
            # Create validation split from training data for early stopping
            from sklearn.model_selection import train_test_split
            X_t, X_v, y_t, y_v = train_test_split(
                X_train, y_train, test_size=0.15, random_state=self.random_state
            )
            model.fit(
                X_t, y_t,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        
        self.models['lgbm_regression'] = model
        print("✓ LightGBM training complete")
        
        return model
    
    def train_xgb_classifier(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None, 
        params: Optional[Dict] = None
    ):
        """
        Train XGBoost classifier (primary classification model).
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model parameters
        
        Returns:
            Trained model
        """
        print("Training XGBoost Classifier...")

        if params is None:
            params = self.get_default_params('xgb', 'classification')

        model = xgb.XGBClassifier(**params)

        # Helper to call fit with or without early stopping depending on installed xgboost
        from inspect import signature
        def _safe_fit(m, X_tr, y_tr, X_va=None, y_va=None, es_rounds: int = 100):
            sig = signature(m.fit)
            fit_kwargs = {}
            if X_va is not None and y_va is not None:
                # include eval_set when validation data provided
                fit_kwargs['eval_set'] = [(X_va, y_va)]

            # Only pass early_stopping_rounds if fit accepts it
            if 'early_stopping_rounds' in sig.parameters:
                fit_kwargs['early_stopping_rounds'] = es_rounds

            # Pass verbose if supported
            if 'verbose' in sig.parameters:
                fit_kwargs['verbose'] = False

            try:
                return m.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                # Older/newer xgboost scikit wrapper may not accept these kwargs; fall back
                return m.fit(X_tr, y_tr)

        try:
            if X_val is not None and y_val is not None:
                _safe_fit(model, X_train, y_train, X_val, y_val, es_rounds=100)
            else:
                # Create validation split from training data for early stopping
                from sklearn.model_selection import train_test_split
                try:
                    X_t, X_v, y_t, y_v = train_test_split(
                        X_train, y_train, test_size=0.15, random_state=self.random_state, stratify=y_train
                    )
                except Exception:
                    X_t, X_v, y_t, y_v = train_test_split(
                        X_train, y_train, test_size=0.15, random_state=self.random_state
                    )
                _safe_fit(model, X_t, y_t, X_v, y_v, es_rounds=100)

            self.models['xgb_classification'] = model
            print("✓ XGBoost training complete")
            return model

        except Exception as e:
            print(f"⚠ Warning: Training Classification (Gap Type) failed: {e}")
            return None
    
    def train_baseline_models(
        self,
        X_train,
        y_train,
        task: str = 'regression',
        models_to_train: Optional[list] = None
    ) -> Dict:
        """
        Train multiple baseline models.
        
        Args:
            X_train: Training features
            y_train: Training target
            task: 'regression' or 'classification'
            models_to_train: List of model names to train
        
        Returns:
            Dictionary of trained models
        """
        print(f"\nTraining baseline {task} models...")
        
        if models_to_train is None:
            if task == 'regression':
                models_to_train = ['rf', 'xgb', 'catboost', 'mlp']
            else:
                models_to_train = ['rf', 'lgbm', 'catboost', 'mlp']
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                print(f"\n  Training {model_name.upper()}...")
                model = self.get_model(model_name, task)
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                self.models[f"{model_name}_{task}"] = model
                print(f"  ✓ {model_name.upper()} trained")
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
        
        print(f"\n✓ Trained {len(trained_models)} baseline models")
        return trained_models
    
    def cross_validate_model(
        self,
        model,
        X,
        y,
        cv: int = 5,
        scoring: Optional[list] = None
    ) -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of CV folds
            scoring: List of scoring metrics
        
        Returns:
            Dictionary with CV scores
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        if scoring is None:
            scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Compute mean and std for each metric
        results = {}
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in cv_results:
                results[f'{metric}_mean'] = cv_results[test_key].mean()
                results[f'{metric}_std'] = cv_results[test_key].std()
                
            if train_key in cv_results:
                results[f'{metric}_train_mean'] = cv_results[train_key].mean()
        
        print("✓ Cross-validation complete")
        
        return results
    
    def hyperparameter_search(
        self,
        model_name: str,
        X_train,
        y_train,
        task: str = 'regression',
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ):
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            task: 'regression' or 'classification'
            param_grid: Parameter grid (uses default if None)
            cv: Number of CV folds
        
        Returns:
            Best model from grid search
        """
        print(f"\nPerforming hyperparameter search for {model_name}...")
        
        if param_grid is None:
            param_grid = self.get_default_param_grid(model_name, task)
        
        base_model = self.get_model(model_name, task, params={})
        
        scoring = 'neg_mean_absolute_error' if task == 'regression' else 'accuracy'
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best score: {grid_search.best_score_:.4f}")
        
        self.best_params[model_name] = grid_search.best_params_
        
        return grid_search.best_estimator_
    
    def get_default_param_grid(self, model_name: str, task: str) -> Dict:
        """
        Get default parameter grid for hyperparameter search.
        
        Args:
            model_name: Name of the model
            task: 'regression' or 'classification'
        
        Returns:
            Parameter grid dictionary
        """
        grids = {
            'lgbm': {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.03, 0.05],
                'num_leaves': [31, 64],
                'max_depth': [-1, 8],
            },
            'xgb': {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [3, 6, 9],
            },
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
            }
        }
        
        return grids.get(model_name, {})
    
    def save_model(self, model_name: str, output_dir: str, metadata: Optional[Dict] = None):
        """
        Save trained model.
        
        Args:
            model_name: Name of the model
            output_dir: Directory to save model
            metadata: Optional metadata dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(self.models[model_name], model_path)
        print(f"✓ Model saved to {model_path}")
        
        if metadata:
            import json
            meta_path = output_dir / f"{model_name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"✓ Metadata saved to {meta_path}")


def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: List[str],
    output_dir: str = 'models',
    task: str = 'regression'
) -> Dict:
    """
    Train multiple models and save them.
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training target (Series or array)
        X_test: Test features (DataFrame or array)
        y_test: Test target (Series or array)
        feature_names: List of feature names
        output_dir: Directory to save models
        task: 'regression' or 'classification'
    
    Returns:
        Dictionary with trained model information
    """
    print("\n" + "="*80)
    print(f"MODEL TRAINING - {task.upper()}")
    print("="*80 + "\n")
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train models based on task
    if task == 'regression':
        print("\n1. Training LightGBM (primary)...")
        lgbm_model = trainer.train_lgbm_regression(X_train, y_train, X_test, y_test)
        
        print("\n2. Training baseline models...")
        baseline_models = trainer.train_baseline_models(X_train, y_train, task='regression')
        
    else:  # classification
        print("\n1. Training XGBoost Classifier (primary)...")
        xgb_model = trainer.train_xgb_classifier(X_train, y_train, X_test, y_test)
        
        print("\n2. Training baseline models...")
        baseline_models = trainer.train_baseline_models(X_train, y_train, task='classification')
    
    # Save models
    print(f"\nSaving models to {output_dir}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trained_models = {}
    for model_name, model in trainer.models.items():
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✓ {model_name} saved to {model_path}")
        
        trained_models[model_name] = {
            'model': model,
            'path': str(model_path)
        }
    
    # Save feature names
    feature_names_path = output_dir / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    print(f"✓ Feature names saved to {feature_names_path}")
    
    print("\n✓ Model training complete!")
    
    return trained_models


def train_models_from_paths(
    X_train_path: str,
    y_train_path: str,
    X_test_path: str,
    y_test_path: str,
    task: str = 'regression',
    output_dir: str = 'models'
) -> Dict:
    """
    Main function to train all models from saved data paths.
    
    Args:
        X_train_path: Path to training features
        y_train_path: Path to training target
        X_test_path: Path to test features
        y_test_path: Path to test target
        task: 'regression' or 'classification'
        output_dir: Directory to save models
    
    Returns:
        Dictionary with trained models
    """
    print("\n" + "="*80)
    print(f"MODEL TRAINING - {task.upper()}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading preprocessed data...")
    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Get feature names
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # Call the main train_models function
    return train_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
        task=task
    )


# Keep the old function name for backwards compatibility
def train_models_old(
    X_train_path: str,
    y_train_path: str,
    X_test_path: str,
    y_test_path: str,
    task: str = 'regression',
    output_dir: str = 'models'
) -> Dict:
    """
    Legacy function - use train_models_from_paths instead.
    
    Args:
        X_train_path: Path to training features
        y_train_path: Path to training target
        X_test_path: Path to test features
        y_test_path: Path to test target
        task: 'regression' or 'classification'
        output_dir: Directory to save models
    
    Returns:
        Dictionary with trained models
    """
    print("\n" + "="*80)
    print(f"MODEL TRAINING - {task.upper()}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading preprocessed data...")
    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train primary model
    if task == 'regression':
        print("\nTraining primary regression model (LightGBM)...")
        primary_model = trainer.train_lgbm_regression(X_train, y_train)
    else:
        print("\nTraining primary classification model (XGBoost)...")
        primary_model = trainer.train_xgb_classifier(X_train, y_train)
    
    # Train baseline models
    baseline_models = trainer.train_baseline_models(X_train, y_train, task=task)
    
    # Save models
    print(f"\nSaving models to {output_dir}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in trainer.models.items():
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"✓ {model_name} saved")
    
    print("\n✓ Model training complete!")
    
    return {
        'trainer': trainer,
        'primary_model': primary_model,
        'baseline_models': baseline_models
    }


if __name__ == "__main__":
    # Test model training
    print("Testing model training module...")
    
    try:
        # This requires preprocessed data to exist
        result = train_models(
            X_train_path="data/processed/mean_regression_X_train.pkl",
            y_train_path="data/processed/mean_regression_y_train.pkl",
            X_test_path="data/processed/mean_regression_X_test.pkl",
            y_test_path="data/processed/mean_regression_y_test.pkl",
            task='regression'
        )
        
        print("\n✓ Model training test successful")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Note: This test requires preprocessed data files to exist")
