import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv('train_meta.csv')
feats = pd.read_csv('pipeline_v2/train_features.csv.gz')

labels = [
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
label_to_idx = {l: i for i, l in enumerate(labels)}

feature_cols = [c for c in feats.columns if c not in ['id', 'label']]
X = feats[feature_cols].copy()
y = feats['label'].map(label_to_idx).values

X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
    X, y, feats.index, test_size=0.2, stratify=y, random_state=42
)

# Train LightGBM on train split only
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_tr, y_tr)
probs = model.predict_proba(X_va)
preds = np.argmax(probs, axis=1)

print("="*60)
print("ERROR AUDIT PER CLASS ON 80/20 VALIDATION SPLIT")
print("="*60)

cm = confusion_matrix(y_va, preds)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion Matrix (Rows: True, Cols: Pred):")
print(df_cm.to_string())

# For each class, find the primary confusion target
print("\nPrimary Error Modes:")
for i, l in enumerate(labels):
    row = df_cm.iloc[i].copy()
    correct = row[l]
    total = row.sum()
    row[l] = 0
    top_err_class = row.idxmax()
    top_err_cnt = row.max()
    print(f"  {l:<18}: {correct}/{total} correct ({correct/total*100:5.1f}%). Main error: predicted as {top_err_class} ({top_err_cnt} times)")
