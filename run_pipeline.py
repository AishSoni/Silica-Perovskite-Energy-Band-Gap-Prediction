"""
Main pipeline script for perovskite bandgap prediction.
Orchestrates the entire ML workflow from data loading to evaluation.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import set_seeds, print_section, save_json, get_system_info
from src.data_io import load_and_prepare_data
from src.featurize import featurize_data
from src.preprocess import preprocess_data
from src.models import train_models
from src.eval import evaluate_model


def main():
    """
    Run the complete ML pipeline.
    """
    # Set random seeds for reproducibility
    set_seeds(42)
    
    print_section("PEROVSKITE BANDGAP PREDICTION PIPELINE", "=", 100)
    
    # Save system information for reproducibility
    print("\nRecording system information...")
    system_info = get_system_info()
    save_json(system_info, "experiments/system_info.json")
    
    # Step 1: Load and prepare data
    print_section("STEP 1: DATA LOADING AND PREPARATION", "-", 100)
    try:
        df_raw, metadata = load_and_prepare_data(
            data_dir="materials_data",
            output_dir="data/raw"
        )
        print("✓ Step 1 complete")
    except Exception as e:
        print(f"✗ Error in Step 1: {e}")
        return
    
    # Step 2: Feature engineering
    print_section("STEP 2: FEATURE ENGINEERING", "-", 100)
    try:
        df_features = featurize_data(
            input_path="data/raw/perovskites_raw.csv",
            output_path="data/processed/perovskites_features.csv",
            feature_list_path="data/processed/features_list.csv"
        )
        print("✓ Step 2 complete")
    except Exception as e:
        print(f"✗ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Preprocessing - Multiple variants
    print_section("STEP 3: DATA PREPROCESSING", "-", 100)
    
    preprocessing_configs = [
        # Regression with mean imputation
        {
            'imputation_strategy': 'mean',
            'remove_metals': False,
            'for_classification': False,
            'name': 'regression_all_mean'
        },
        # Regression with KNN imputation
        {
            'imputation_strategy': 'knn',
            'remove_metals': False,
            'for_classification': False,
            'name': 'regression_all_knn'
        },
        # Regression without metals (mean)
        {
            'imputation_strategy': 'mean',
            'remove_metals': True,
            'for_classification': False,
            'name': 'regression_nonmetals_mean'
        },
        # Classification (direct vs indirect)
        {
            'imputation_strategy': 'mean',
            'remove_metals': False,
            'for_classification': True,
            'name': 'classification_mean'
        }
    ]
    
    for config in preprocessing_configs:
        print(f"\n--- Preprocessing: {config['name']} ---")
        try:
            preprocess_data(
                input_path="data/processed/perovskites_features.csv",
                output_dir="data/processed",
                imputation_strategy=config['imputation_strategy'],
                remove_metals=config['remove_metals'],
                for_classification=config['for_classification']
            )
        except Exception as e:
            print(f"⚠ Warning: Preprocessing {config['name']} failed: {e}")
            continue
    
    print("\n✓ Step 3 complete")
    
    # Step 4: Model training
    print_section("STEP 4: MODEL TRAINING", "-", 100)
    
    # Train regression models
    training_configs = [
        {
            'prefix': 'mean_regression_',
            'task': 'regression',
            'name': 'Regression (All, Mean Imputation)'
        },
        {
            'prefix': 'knn_regression_',
            'task': 'regression',
            'name': 'Regression (All, KNN Imputation)'
        },
        {
            'prefix': 'mean_regression_nonmetals_',
            'task': 'regression',
            'name': 'Regression (Non-metals, Mean)'
        },
        {
            'prefix': 'mean_classification_',
            'task': 'classification',
            'name': 'Classification (Gap Type)'
        }
    ]
    
    for config in training_configs:
        print(f"\n--- Training: {config['name']} ---")
        try:
            train_models(
                X_train_path=f"data/processed/{config['prefix']}X_train.pkl",
                y_train_path=f"data/processed/{config['prefix']}y_train.pkl",
                X_test_path=f"data/processed/{config['prefix']}X_test.pkl",
                y_test_path=f"data/processed/{config['prefix']}y_test.pkl",
                task=config['task'],
                output_dir='models'
            )
        except Exception as e:
            print(f"⚠ Warning: Training {config['name']} failed: {e}")
            continue
    
    print("\n✓ Step 4 complete")
    
    # Step 5: Model evaluation
    print_section("STEP 5: MODEL EVALUATION", "-", 100)
    
    evaluation_configs = [
        {
            'model_path': 'models/lgbm_regression.pkl',
            'prefix': 'mean_regression_',
            'task': 'regression',
            'output_dir': 'figures/regression_all_mean',
            'name': 'LightGBM Regression (All, Mean)'
        },
        {
            'model_path': 'models/lgbm_regression.pkl',
            'prefix': 'mean_regression_nonmetals_',
            'task': 'regression',
            'output_dir': 'figures/regression_nonmetals_mean',
            'name': 'LightGBM Regression (Non-metals)'
        },
        {
            'model_path': 'models/xgb_classification.pkl',
            'prefix': 'mean_classification_',
            'task': 'classification',
            'output_dir': 'figures/classification',
            'name': 'XGBoost Classification'
        }
    ]
    
    results_summary = {}
    
    for config in evaluation_configs:
        print(f"\n--- Evaluating: {config['name']} ---")
        try:
            metrics = evaluate_model(
                model_path=config['model_path'],
                X_test_path=f"data/processed/{config['prefix']}X_test.pkl",
                y_test_path=f"data/processed/{config['prefix']}y_test.pkl",
                task=config['task'],
                output_dir=config['output_dir'],
                feature_names_path=f"data/processed/{config['prefix']}feature_names.txt"
            )
            results_summary[config['name']] = metrics
        except Exception as e:
            print(f"⚠ Warning: Evaluation {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results summary
    if results_summary:
        save_json(results_summary, "results/all_models_summary.json")
        print("\n✓ Results summary saved to results/all_models_summary.json")
    
    print("\n✓ Step 5 complete")
    
    # Final summary
    print_section("PIPELINE COMPLETE", "=", 100)
    print("\nOutputs:")
    print("  - Raw data: data/raw/")
    print("  - Features: data/processed/perovskites_features.csv")
    print("  - Models: models/")
    print("  - Figures: figures/")
    print("  - Results: results/")
    print("\nNext steps:")
    print("  1. Review figures in figures/ directory")
    print("  2. Check model performance in results/all_models_summary.json")
    print("  3. Explore notebooks/MinimalPipeline.ipynb for interactive analysis")
    print("  4. Write paper using templates in paper/")
    

if __name__ == "__main__":
    main()
