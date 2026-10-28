import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def generate_v3_submission(
    features_csv='pipeline_v2/test_features_v3.csv.gz',
    sample_sub_csv='sample_submission.csv',
    output_csv='pipeline_v2/submission_v3.csv',
    root_output_csv='submission.csv'
):
    print("="*65)
    print("GENERATING FINAL V3 HYBRID COMPETITION SUBMISSION")
    print("="*65)
    
    with open('pipeline_v2/config_v3.json', 'r') as f:
        config = json.load(f)
        
    labels = config['labels']
    feature_cols = config['feature_cols']
    w_lgb, w_xgb, w_cnn = config['blend_weights']
    class_weights = np.array(config['class_weights'], dtype=np.float32)
    
    print(f"Loading test features from {features_csv}...")
    df_test = pd.read_csv(features_csv)
    X_test = df_test[feature_cols].copy()
    test_ids = df_test['id'].values
    print(f"Test samples: {len(X_test)}, Features: {X_test.shape[1]}")
    
    print("Loading final GBDT models...")
    lgb_m = joblib.load('pipeline_v2/final_lgb_v3.joblib')
    xgb_m = joblib.load('pipeline_v2/final_xgb_v3.joblib')
    
    p_lgb = lgb_m.predict_proba(X_test)
    p_xgb = xgb_m.predict_proba(X_test)
    
    # Check if CNN test probs are ready
    if os.path.exists('pipeline_v2/cnn_test_probs.npy') and w_cnn > 0:
        print("Loading PyTorch SpectrogramNet test probabilities...")
        p_cnn = np.load('pipeline_v2/cnn_test_probs.npy')
        print(f"Applying Hybrid Blend: {w_lgb:.2f} LGB + {w_xgb:.2f} XGB + {w_cnn:.2f} CNN")
        p_ens = w_lgb * p_lgb + w_xgb * p_xgb + w_cnn * p_cnn
    else:
        print("Blending GBDT models (0.55 LGB + 0.45 XGB)...")
        p_ens = 0.55 * p_lgb + 0.45 * p_xgb
        
    # Save test probabilities
    np.save('pipeline_v2/test_probabilities_v3.npy', p_ens)
    
    # Apply calibrated decision weights
    p_calib = p_ens * class_weights
    pred_indices = np.argmax(p_calib, axis=1)
    pred_labels = [labels[idx] for idx in pred_indices]
    
    sub_df = pd.DataFrame({
        'id': test_ids,
        'label': pred_labels
    })
    
    # 10 INTEGRITY CHECKS
    print("\n" + "-"*50)
    print("VERIFYING SUBMISSION INTEGRITY...")
    print("-"*50)
    sample_sub = pd.read_csv(sample_sub_csv)
    
    assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
    print(f"  [PASS] Exactly 10,000 test predictions.")
    assert list(sub_df.columns) == ['id', 'label'], f"Columns mismatch: {list(sub_df.columns)}"
    print(f"  [PASS] Schema matches 'id,label'.")
    assert (sub_df['id'].values == sample_sub['id'].values).all(), "ID alignment mismatch!"
    print(f"  [PASS] IDs match sample_submission.csv 100% in exact order.")
    assert sub_df['id'].isna().sum() == 0, "Missing IDs!"
    print(f"  [PASS] No missing IDs.")
    assert sub_df['id'].nunique() == len(sub_df), "Duplicate IDs!"
    print(f"  [PASS] All IDs are unique.")
    assert sub_df['label'].isna().sum() == 0, "Missing predictions!"
    print(f"  [PASS] No null or NaN predictions.")
    invalid = set(sub_df['label']) - set(labels)
    assert len(invalid) == 0, f"Invalid classes: {invalid}"
    print(f"  [PASS] All 9 labels valid.")
    
    pred_counts = sub_df['label'].value_counts()
    print("\nTest Prediction Distribution:")
    for l in labels:
        cnt = pred_counts.get(l, 0)
        pct = cnt / len(sub_df) * 100
        print(f"  {l:<20}: {cnt:5d} ({pct:5.2f}%)")
        assert cnt > 0, f"Error: class {l} has 0 predictions!"
    print("  [PASS] All 9 classes actively predicted.")
    
    sub_df.to_csv(output_csv, index=False)
    print(f"\nSaved v3 submission to: {output_csv}")
    
    if root_output_csv:
        sub_df.to_csv(root_output_csv, index=False)
        print(f"Updated root submission file: {root_output_csv}")
        
    print("\nSubmission ready for upload!")

if __name__ == '__main__':
    generate_v3_submission()
