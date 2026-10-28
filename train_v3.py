import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

LABELS = [
    'eti_candidate',
    'frb',
    'noise',
    'pulsar',
    'radar_chirp',
    'rfi_intermittent',
    'rfi_persistent',
    'satellite_pass',
    'scintillating_star'
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}

def optimize_thresholds(y_true, probs, base_macro_f1):
    best_weights = np.ones(len(LABELS), dtype=np.float32)
    best_f1 = base_macro_f1
    candidates = [0.8, 1.0, 1.25, 1.5, 1.8, 2.2, 2.6]
    
    # Optimize coordinates: eti, frb, scintillating_star, satellite_pass, rfi_intermittent
    for cls_idx in [0, 8, 7, 1, 5, 4]:
        best_c_w = best_weights[cls_idx]
        for w in candidates:
            test_weights = best_weights.copy()
            test_weights[cls_idx] = w
            scaled_probs = probs * test_weights
            preds = np.argmax(scaled_probs, axis=1)
            score = f1_score(y_true, preds, average='macro')
            if score > best_f1:
                best_f1 = score
                best_c_w = w
        best_weights[cls_idx] = best_c_w
    return best_weights, best_f1

def train_and_benchmark_v3():
    print("="*65)
    print("HACK4HACK 2026: V3 BENCHMARK & MULTI-MODEL HYBRID ENSEMBLE")
    print("="*65)
    
    data_path = 'pipeline_v2/train_features_v3.csv.gz'
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ['id', 'label']]
    X = df[feature_cols].copy()
    y = df['label'].map(LABEL_TO_IDX).values
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # 80/20 Stratified Split
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        X, y, df.index, test_size=0.20, stratify=y, random_state=42
    )
    
    class_counts = np.bincount(y_tr, minlength=len(LABELS))
    class_weights_dict = {i: len(y_tr) / (len(LABELS) * class_counts[i]) for i in range(len(LABELS))}
    sample_weights_tr = np.array([class_weights_dict[yi] for yi in y_tr])
    
    results = []
    
    # 1. LightGBM v3
    print("\n--- Training Model A: LightGBM (v3 features) ---")
    t0 = time.time()
    lgb_m = lgb.LGBMClassifier(
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=35,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_m.fit(X_tr, y_tr)
    t_lgb = time.time() - t0
    lgb_probs = lgb_m.predict_proba(X_va)
    lgb_f1 = f1_score(y_va, np.argmax(lgb_probs, axis=1), average='macro')
    print(f"Model A (LightGBM v3) Val Macro-F1: {lgb_f1:.4f} ({t_lgb:.1f}s)")
    results.append({'model': 'LightGBM v3', 'val_macro_f1': round(lgb_f1, 4), 'train_time_s': round(t_lgb, 1)})
    
    # 2. XGBoost v3
    print("\n--- Training Model B: XGBoost (v3 features) ---")
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='multi:softprob',
        num_class=len(LABELS),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_m.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    t_xgb = time.time() - t0
    xgb_probs = xgb_m.predict_proba(X_va)
    xgb_f1 = f1_score(y_va, np.argmax(xgb_probs, axis=1), average='macro')
    print(f"Model B (XGBoost v3) Val Macro-F1: {xgb_f1:.4f} ({t_xgb:.1f}s)")
    results.append({'model': 'XGBoost v3', 'val_macro_f1': round(xgb_f1, 4), 'train_time_s': round(t_xgb, 1)})
    
    # 3. Load CNN Val Probs if available
    cnn_probs = None
    if os.path.exists('pipeline_v2/cnn_val_probs.npy'):
        cnn_probs = np.load('pipeline_v2/cnn_val_probs.npy')
        cnn_f1 = f1_score(y_va, np.argmax(cnn_probs, axis=1), average='macro')
        print(f"\nModel C: PyTorch ON/OFF SpectrogramNet Val Macro-F1: {cnn_f1:.4f}")
        results.append({'model': 'PyTorch SpectrogramNet', 'val_macro_f1': round(cnn_f1, 4), 'train_time_s': 45.0})
        
    # 4. Hybrid Ensemble Optimization
    print("\n--- Evaluating Hybrid Ensembles ---")
    if cnn_probs is not None:
        best_ens_f1 = 0.0
        best_w = (0.45, 0.35, 0.20)
        for w_lgb in [0.4, 0.45, 0.5]:
            for w_xgb in [0.3, 0.35, 0.4]:
                w_cnn = 1.0 - w_lgb - w_xgb
                if w_cnn < 0.1:
                    continue
                ens_p = w_lgb * lgb_probs + w_xgb * xgb_probs + w_cnn * cnn_probs
                score = f1_score(y_va, np.argmax(ens_p, axis=1), average='macro')
                if score > best_ens_f1:
                    best_ens_f1 = score
                    best_w = (w_lgb, w_xgb, w_cnn)
        w1, w2, w3 = best_w
        final_val_probs = w1 * lgb_probs + w2 * xgb_probs + w3 * cnn_probs
        print(f"Optimal Hybrid Weights: LGB={w1:.2f}, XGB={w2:.2f}, CNN={w3:.2f}")
        print(f"Raw Hybrid Ensemble Macro-F1: {best_ens_f1:.4f}")
    else:
        final_val_probs = 0.55 * lgb_probs + 0.45 * xgb_probs
        best_ens_f1 = f1_score(y_va, np.argmax(final_val_probs, axis=1), average='macro')
        print(f"Raw GBDT Ensemble Macro-F1: {best_ens_f1:.4f}")
        best_w = (0.55, 0.45, 0.0)
        
    # Threshold Calibration
    class_weights, calib_f1 = optimize_thresholds(y_va, final_val_probs, best_ens_f1)
    print(f"Calibrated Ensemble Macro-F1: {calib_f1:.4f}")
    results.append({'model': 'Calibrated Hybrid Ensemble', 'val_macro_f1': round(calib_f1, 4), 'train_time_s': round(t_lgb + t_xgb, 1)})
    
    calib_preds = np.argmax(final_val_probs * class_weights, axis=1)
    rep_str = classification_report(y_va, calib_preds, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_va, calib_preds)
    
    print("\n" + "="*60)
    print("FINAL VALIDATION CLASSIFICATION REPORT")
    print("="*60)
    print(rep_str)
    print("Confusion Matrix:")
    print(cm)
    
    # Save validation report
    with open('pipeline_v2/validation_report_v3.txt', 'w') as f:
        f.write("HACK4HACK 2026: SIGNAL HUNTERS ROUND 2 - V3 BENCHMARK REPORT\n")
        f.write("="*65 + "\n\n")
        f.write(f"Starter Baseline LB: 0.4757 / Val: 0.5429\n")
        f.write(f"Best Hybrid Model Validation Macro-F1: {calib_f1:.4f}\n\n")
        f.write("EXPERIMENT RESULTS:\n")
        f.write(pd.DataFrame(results).to_string(index=False) + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(rep_str + "\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(str(cm) + "\n")
        
    # Save artifacts
    artifacts = {
        'blend_weights': [float(x) for x in best_w],
        'class_weights': [float(x) for x in class_weights],
        'labels': LABELS,
        'feature_cols': feature_cols
    }
    with open('pipeline_v2/config_v3.json', 'w') as f:
        json.dump(artifacts, f, indent=2)
        
    # Retrain final models on 100% of data
    print("\nRetraining final GBDT models on 100% data...")
    full_sample_weights = np.array([
        (len(y) / (len(LABELS) * np.bincount(y, minlength=len(LABELS))[yi])) for yi in y
    ])
    final_lgb = lgb.LGBMClassifier(
        n_estimators=450, learning_rate=0.035, num_leaves=35, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    final_lgb.fit(X, y)
    
    final_xgb = xgb.XGBClassifier(
        n_estimators=350, learning_rate=0.04, max_depth=5,
        subsample=0.85, colsample_bytree=0.85, objective='multi:softprob',
        num_class=len(LABELS), random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    final_xgb.fit(X, y, sample_weight=full_sample_weights)
    
    joblib.dump(final_lgb, 'pipeline_v2/final_lgb_v3.joblib')
    joblib.dump(final_xgb, 'pipeline_v2/final_xgb_v3.joblib')
    print("Saved final models to pipeline_v2/final_*_v3.joblib and config_v3.json")

if __name__ == '__main__':
    train_and_benchmark_v3()
