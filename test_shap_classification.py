"""
Test script to verify SHAP and classification functionality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import set_seeds
from src.data_io import prepare_training_data
from src.preprocess import split_and_scale_data
from src.models import train_models
from src.eval import evaluate_model, ModelEvaluator
import joblib

def test_regression_with_shap():
    """Test regression with SHAP analysis."""
    print("\n" + "="*80)
    print("TEST 1: REGRESSION WITH SHAP (F10)")
    print("="*80 + "\n")
    
    set_seeds(42)
    
    # Load data
    X, y, feature_names = prepare_training_data(
        subset_name="F10",
        task='regression'
    )
    
    # Split data
    split_data = split_and_scale_data(X, y, feature_names, test_size=0.2, random_state=42)
    
    # Train only LightGBM (quick test)
    print("\nTraining LightGBM...")
    from src.models import ModelTrainer
    trainer = ModelTrainer(random_state=42)
    model = trainer.train_lgbm_regression(split_data['X_train'], split_data['y_train'])
    
    # Evaluate
    print("\nEvaluating model...")
    output_dir = Path("test_output/regression_shap")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = evaluate_model(
        model=model,
        X_test=split_data['X_test'],
        y_test=split_data['y_test'],
        task='regression',
        output_dir=str(output_dir),
        model_name='lgbm_test',
        feature_names=feature_names
    )
    
    print(f"\nMetrics: R²={metrics.get('r2', 0):.4f}, MAE={metrics.get('mae', 0):.4f}")
    
    # SHAP analysis
    print("\nRunning SHAP analysis...")
    evaluator = ModelEvaluator(output_dir=str(output_dir))
    X_shap = split_data['X_test'].head(100)  # Use 100 samples
    
    evaluator.shap_analysis(
        model=model,
        X=X_shap,
        feature_names=feature_names,
        max_display=10,
        prefix="shap_test"
    )
    
    print(f"\n✓ Test 1 complete! Check {output_dir} for outputs")


def test_classification():
    """Test classification task."""
    print("\n" + "="*80)
    print("TEST 2: CLASSIFICATION (BANDGAP TYPE)")
    print("="*80 + "\n")
    
    set_seeds(42)
    
    try:
        # Load data for classification
        X, y, feature_names = prepare_training_data(
            subset_name="F10",
            task='classification'
        )
        
        # Split data
        split_data = split_and_scale_data(X, y, feature_names, test_size=0.2, random_state=42)
        
        # Train XGBoost classifier
        print("\nTraining XGBoost Classifier...")
        from src.models import ModelTrainer
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_xgb_classifier(split_data['X_train'], split_data['y_train'])
        
        if model is not None:
            # Evaluate
            print("\nEvaluating model...")
            output_dir = Path("test_output/classification")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metrics = evaluate_model(
                model=model,
                X_test=split_data['X_test'],
                y_test=split_data['y_test'],
                task='classification',
                output_dir=str(output_dir),
                model_name='xgb_classifier_test',
                feature_names=feature_names
            )
            
            print(f"\nMetrics: Accuracy={metrics.get('Accuracy', 0):.4f}, F1={metrics.get('F1-Score', 0):.4f}, Precision={metrics.get('Precision', 0):.4f}, Recall={metrics.get('Recall', 0):.4f}")
            
            # Run SHAP analysis
            print("\nRunning SHAP analysis for classification...")
            evaluator = ModelEvaluator(output_dir=str(output_dir))
            evaluator.shap_analysis(
                model=model,
                X=split_data['X_test'][:100],  # Use subset for speed
                feature_names=feature_names,
                max_display=10,
                prefix="shap_classifier"
            )
            
            print(f"\n✓ Test 2 complete! Check {output_dir} for outputs")
        else:
            print("\n⚠ Classification model training failed - likely missing is_gap_direct column")
            
    except Exception as e:
        print(f"\n⚠ Classification test failed: {e}")
        print("This is expected if 'is_gap_direct' column is not in the dataset")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING ENHANCED PIPELINE FEATURES")
    print("="*80)
    
    # Test 1: Regression with SHAP
    try:
        test_regression_with_shap()
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Classification
    try:
        test_classification()
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
