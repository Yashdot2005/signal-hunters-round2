import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import sys

sys.path.insert(0, os.path.abspath('pipeline_v2'))
from cnn_model_v4 import CadenceDatasetV4, CadenceNetV4, LABELS, LABEL_TO_IDX

def train_cnn_v4(epochs=18, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"TRAINING ENHANCED CadenceNetV4 ON: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    print("Loading cadence arrays and metadata...")
    X_train = np.load('train_cadences.npy', mmap_mode='r')
    meta_train = pd.read_csv('train_meta.csv')
    y = meta_train['label'].map(LABEL_TO_IDX).values
    
    # 80/20 Stratified Split
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=42
    )
    
    # Class weights with smoothing
    class_counts = np.bincount(y[idx_tr], minlength=len(LABELS))
    # Moderate power weighting to prevent overfitting rare classes
    raw_weights = len(idx_tr) / (len(LABELS) * class_counts.astype(np.float32))
    norm_weights = (raw_weights / np.mean(raw_weights)) ** 0.8
    class_weights_t = torch.from_numpy(norm_weights).float().to(device)
    
    train_dataset = CadenceDatasetV4(X_train[idx_tr], meta_train.iloc[idx_tr], y[idx_tr], is_train=True)
    val_dataset = CadenceDatasetV4(X_train[idx_va], meta_train.iloc[idx_va], y[idx_va], is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = CadenceNetV4(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_macro_f1 = 0.0
    best_val_probs = None
    
    print(f"\nStarting CadenceNetV4 Training ({epochs} Epochs)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch_cad, batch_diff, batch_meta, batch_y in train_loader:
            batch_cad = batch_cad.to(device)
            batch_diff = batch_diff.to(device)
            batch_meta = batch_meta.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_cad, batch_diff, batch_meta)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_probs_list = []
        val_targets = []
        with torch.no_grad():
            for batch_cad, batch_diff, batch_meta, batch_y in val_loader:
                batch_cad = batch_cad.to(device)
                batch_diff = batch_diff.to(device)
                batch_meta = batch_meta.to(device)
                logits = model(batch_cad, batch_diff, batch_meta)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs_list.append(probs)
                val_targets.extend(batch_y.numpy())
                
        val_probs = np.vstack(val_probs_list)
        val_preds = np.argmax(val_probs, axis=1)
        val_targets = np.array(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} - Train Loss: {train_loss:.4f} - Val Macro-F1: {val_f1:.4f} ({dt:.1f}s)")
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_val_probs = val_probs
            torch.save(model.state_dict(), 'pipeline_v2/best_cnn_v4.pt')
            np.save('pipeline_v2/cnn_v4_val_probs.npy', val_probs)
            
    print(f"\nBest CadenceNetV4 Validation Macro-F1: {best_macro_f1:.4f}")
    
    # Generate test predictions
    print("\nGenerating Test Predictions with Best CadenceNetV4...")
    X_test = np.load('test_cadences.npy', mmap_mode='r')
    meta_test = pd.read_csv('test_meta.csv')
    test_dataset = CadenceDatasetV4(X_test, meta_test, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.load_state_dict(torch.load('pipeline_v2/best_cnn_v4.pt'))
    model.eval()
    test_probs_list = []
    with torch.no_grad():
        for batch_cad, batch_diff, batch_meta in test_loader:
            batch_cad = batch_cad.to(device)
            batch_diff = batch_diff.to(device)
            batch_meta = batch_meta.to(device)
            logits = model(batch_cad, batch_diff, batch_meta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_probs_list.append(probs)
            
    test_probs = np.vstack(test_probs_list)
    np.save('pipeline_v2/cnn_v4_test_probs.npy', test_probs)
    print(f"Saved test probabilities to pipeline_v2/cnn_v4_test_probs.npy ({test_probs.shape})")

if __name__ == '__main__':
    train_cnn_v4()
