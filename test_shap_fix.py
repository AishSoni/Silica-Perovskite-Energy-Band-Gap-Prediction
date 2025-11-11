"""
Quick test script to verify XGBoost SHAP fix works
Run this after retraining models to confirm SHAP analysis works
"""

import joblib
import shap
import pandas as pd
import numpy as np

print("="*80)
print("TESTING XGBOOST SHAP FIX")
print("="*80)

try:
    # Load model and test data
    print("\n1. Loading XGBoost classification model...")
    model = joblib.load('models/xgb_classification.pkl')
    print("   ✓ Model loaded")
    
    print("\n2. Loading test data...")
    X_test = joblib.load('data/processed/mean_classification_X_test.pkl')
    print(f"   ✓ Test data loaded: {X_test.shape}")
    
    print("\n3. Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("   ✓ Explainer created successfully")
    
    print("\n4. Computing SHAP values (sample of 10)...")
    shap_values = explainer.shap_values(X_test.iloc[:10])
    print(f"   ✓ SHAP values computed: shape {np.array(shap_values).shape}")
    
    print("\n5. Testing SHAP visualization...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    shap.summary_plot(shap_values, X_test.iloc[:10], show=False)
    plt.savefig('test_shap_fix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ SHAP plot saved to test_shap_fix.png")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - SHAP FIX WORKS!")
    print("="*80)
    
except ValueError as e:
    if "could not convert string to float" in str(e):
        print("\n" + "="*80)
        print("❌ SHAP FIX NOT APPLIED YET")
        print("="*80)
        print(f"\nError: {e}")
        print("\nThe model still has the base_score string issue.")
        print("Please retrain the model using: python run_pipeline.py")
    else:
        raise
        
except FileNotFoundError as e:
    print("\n" + "="*80)
    print("⚠️  MODEL FILES NOT FOUND")
    print("="*80)
    print(f"\nError: {e}")
    print("\nPlease train the model first using: python run_pipeline.py")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ UNEXPECTED ERROR")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
