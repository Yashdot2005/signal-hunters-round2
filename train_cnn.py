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
from cnn_model import CadenceDataset, OnOffSpectrogramNet, LABELS, LABEL_TO_IDX

def train_cnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CNN on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    print("Loading data...")
    X_train = np.load('train_cadences.npy', mmap_mode='r')
    meta_train = pd.read_csv('train_meta.csv')
    y = meta_train['label'].map(LABEL_TO_IDX).values
    
    # 80/20 Stratified Split
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=42
    )
    
    # Compute class weights
    class_counts = np.bincount(y[idx_tr], minlength=len(LABELS))
    class_weights = len(idx_tr) / (len(LABELS) * class_counts.astype(np.float32))
    class_weights_t = torch.from_numpy(class_weights).float().to(device)
    
    train_dataset = CadenceDataset(X_train[idx_tr], y[idx_tr], is_train=True)
    val_dataset = CadenceDataset(X_train[idx_va], y[idx_va], is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    model = OnOffSpectrogramNet(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    best_macro_f1 = 0.0
    best_val_probs = None
    
    print("\nStarting Training (15 Epochs)...")
    for epoch in range(1, 16):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_probs_list = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                val_probs_list.append(probs)
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        val_probs = np.vstack(val_probs_list)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/15 - Loss: {train_loss:.4f} - Val Macro-F1: {val_f1:.4f} ({dt:.1f}s)")
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_val_probs = val_probs
            torch.save(model.state_dict(), 'pipeline_v2/best_cnn.pt')
            np.save('pipeline_v2/cnn_val_probs.npy', val_probs)
            
    print(f"\nBest CNN Validation Macro-F1: {best_macro_f1:.4f}")
    
    # Generate test predictions
    print("\nPredicting on Test Cadences...")
    X_test = np.load('test_cadences.npy', mmap_mode='r')
    test_dataset = CadenceDataset(X_test, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Load best checkpoint
    model.load_state_dict(torch.load('pipeline_v2/best_cnn.pt'))
    model.eval()
    test_probs_list = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_probs_list.append(probs)
            
    test_probs = np.vstack(test_probs_list)
    np.save('pipeline_v2/cnn_test_probs.npy', test_probs)
    print(f"Saved test probabilities to pipeline_v2/cnn_test_probs.npy ({test_probs.shape})")

if __name__ == '__main__':
    train_cnn()
