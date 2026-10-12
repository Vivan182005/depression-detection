

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.feature_store import FeatureStore
from src.models.depression_model import DualBranchDepressionNet, HAS_TORCH
from src.models.calibrated_model import TemperatureScaler, FocalLoss
from src.models.evaluator import evaluate_research_metrics, plot_research_figures

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler


class LeakFreeDataset(Dataset):
    def __init__(self, master_df, feature_store, scaler=None, is_train=True):
        self.master_df = master_df.reset_index(drop=True)
        self.feature_store = feature_store
        self.scaler = scaler
        self.is_train = is_train

        self.features_list = []
        for idx in range(len(self.master_df)):
            row = self.master_df.iloc[idx]
            f = self.feature_store.load_features(row['session_id'])
            t_mean = np.mean(f['temporal_33d'], axis=0) if len(f['temporal_33d']) > 0 else np.zeros(33)
            f_comb = np.hstack([t_mean, f['egemaps_88d']]) # 121 dims
            bert = f.get('text_bert_768d', np.zeros(768, dtype=np.float32))
            self.features_list.append((f_comb, f['temporal_33d'], f['wav2vec_768d'], bert))

        all_acoustic = np.array([item[0] for item in self.features_list])
        if is_train and self.scaler is not None:
            self.scaler.fit(all_acoustic)

        if self.scaler is not None:
            all_acoustic_scaled = self.scaler.transform(all_acoustic)
        else:
            all_acoustic_scaled = all_acoustic

        self.acoustic_scaled = all_acoustic_scaled

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        row = self.master_df.iloc[idx]
        _, temporal_33d, wav2vec_768d, bert_768d = self.features_list[idx]
        ac_121d = self.acoustic_scaled[idx]

        egemaps_88d = ac_121d[33:]
        temp_pooled = ac_121d[:33]

        return {
            'session_id': row['session_id'],
            'participant_id': row['participant_id'],
            'egemaps': torch.tensor(egemaps_88d, dtype=torch.float32),
            'temporal': torch.tensor(temporal_33d, dtype=torch.float32),
            'wav2vec': torch.tensor(wav2vec_768d, dtype=torch.float32),
            'text_bert': torch.tensor(bert_768d, dtype=torch.float32),
            'class_id': torch.tensor(int(row['severity_class_id']), dtype=torch.long),
            'phq8_score': torch.tensor(float(row['phq8_score']), dtype=torch.float32)
        }


def custom_collate_fn(batch):
    sessions = [item['session_id'] for item in batch]
    pids = [item['participant_id'] for item in batch]
    egemaps = torch.stack([item['egemaps'] for item in batch])
    wav2vec = torch.stack([item['wav2vec'] for item in batch])
    text_bert = torch.stack([item['text_bert'] for item in batch])
    class_ids = torch.stack([item['class_id'] for item in batch])
    phq8_scores = torch.stack([item['phq8_score'] for item in batch])
    
    max_len = max([item['temporal'].shape[0] for item in batch])
    padded_temporal = []
    for item in batch:
        temp = item['temporal']
        if temp.shape[0] < max_len:
            pad = torch.zeros((max_len - temp.shape[0], temp.shape[1]))
            temp = torch.cat([temp, pad], dim=0)
        padded_temporal.append(temp)
    padded_temporal = torch.stack(padded_temporal)

    return {
        'session_id': sessions,
        'participant_id': pids,
        'egemaps': egemaps,
        'temporal': padded_temporal,
        'wav2vec': wav2vec,
        'text_bert': text_bert,
        'class_id': class_ids,
        'phq8_score': phq8_scores
    }


def balance_training_df(train_df):
    """Balanced Random Oversampling of Minority Classes in Training Fold Only."""
    max_samples = train_df['severity_class_id'].value_counts().max()
    dfs = []
    for cls, group in train_df.groupby('severity_class_id'):
        if len(group) < max_samples:
            oversampled = group.sample(max_samples, replace=True, random_state=42)
            dfs.append(oversampled)
        else:
            dfs.append(group)
    balanced_df = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced_df


def run_subject_independent_kfold(n_splits=5, epochs=60, store_dir="./feature_store"):
    print("\n" + "="*80)
    print("   STARTING SUBJECT-INDEPENDENT STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("   Zero Data Leakage | Balanced Oversampling | Temperature Calibration")
    print("="*80)

    feature_store = FeatureStore(store_dir=store_dir)
    master_path = os.path.join(store_dir, "master_dataset_index.csv")
    if not os.path.exists(master_path):
        from label_mapper import build_master_dataset_index
        build_master_dataset_index(store_dir=store_dir)

    master_df = pd.read_csv(master_path)
    print(f"Total Sessions Loaded : {len(master_df)}")
    print("Class Distribution    :")
    print(master_df['severity_category'].value_counts().to_dict())

    sgkf = StratifiedGroupKFold(n_splits=min(n_splits, len(np.unique(master_df['severity_class_id']))))
    groups = master_df['participant_id'].values
    y_class = master_df['severity_class_id'].values

    fold_results = []
    oof_predictions = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(master_df, y_class, groups=groups), 1):
        print(f"\n--------------------------------------------------------------------------")
        print(f"                  EXECUTING FOLD {fold} / {n_splits}")
        print(f"--------------------------------------------------------------------------")

        raw_train_df = master_df.iloc[train_idx]
        val_df = master_df.iloc[val_idx]

        # Balance training fold only (Zero leakage into validation)
        train_df = balance_training_df(raw_train_df)

        scaler = StandardScaler()
        train_dataset = LeakFreeDataset(train_df, feature_store, scaler=scaler, is_train=True)
        val_dataset = LeakFreeDataset(val_df, feature_store, scaler=scaler, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=custom_collate_fn)

        model = DualBranchDepressionNet(num_classes=3, dropout=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['egemaps'], batch['temporal'], batch['wav2vec'], batch['text_bert'])
                
                loss_ce = criterion_ce(outputs['severity_logits'], batch['class_id'])
                loss_mse = criterion_mse(outputs['phq8_score'], batch['phq8_score'])
                loss = loss_ce + 0.1 * loss_mse

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

        # Temperature Scaling Calibration on Validation Set
        calibrated_model = TemperatureScaler(model)
        optimal_T = calibrated_model.calibrate(val_loader)

        # Validation Evaluation
        calibrated_model.eval()
        for batch in val_loader:
            with torch.no_grad():
                out = calibrated_model(batch['egemaps'], batch['temporal'], batch['wav2vec'], batch['text_bert'])
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                phq_pred = out['phq8_score'].numpy()

                true_class = batch['class_id'].numpy()
                true_phq = batch['phq8_score'].numpy()
                s_ids = batch['session_id']

                metrics = evaluate_research_metrics(true_class, preds, probs, true_phq, phq_pred)
                fold_results.append(metrics)
                
                print(f"Fold {fold} Metrics -> Acc: {metrics['accuracy']:.2f}% | Bal Acc: {metrics['balanced_accuracy']:.2f}% | Macro F1: {metrics['f1_macro']:.2f}% | MAE: {metrics['mae']:.2f} | ECE: {metrics['ece']:.2f}%")

                for i in range(len(s_ids)):
                    oof_predictions.append({
                        'fold': fold,
                        'session_id': s_ids[i],
                        'true_class': true_class[i],
                        'pred_class': preds[i],
                        'true_phq': true_phq[i],
                        'pred_phq': phq_pred[i],
                        'confidence': np.max(probs[i]) * 100.0,
                        'calibrated_T': optimal_T
                    })

    # Aggregate Overall Out-of-Sample Metrics across Folds
    oof_df = pd.DataFrame(oof_predictions)
    oof_y_true = oof_df['true_class'].values
    oof_y_pred = oof_df['pred_class'].values
    oof_phq_true = oof_df['true_phq'].values
    oof_phq_pred = oof_df['pred_phq'].values

    # Plot Out-of-Sample Research Figures
    dummy_probs = np.zeros((len(oof_df), 5))
    for i, row in oof_df.iterrows():
        dummy_probs[i, int(row['pred_class'])] = row['confidence'] / 100.0
        
    plot_research_figures(oof_y_true, oof_y_pred, dummy_probs, oof_phq_true, oof_phq_pred, output_dir="./output/plots")

    print("\n" + "="*80)
    print("       OUT-OF-SAMPLE SUBJECT-INDEPENDENT VALIDATION SUMMARY")
    print("="*80)
    
    acc_list = [m['accuracy'] for m in fold_results]
    bal_acc_list = [m['balanced_accuracy'] for m in fold_results]
    f1_macro_list = [m['f1_macro'] for m in fold_results]
    f1_w_list = [m['f1_weighted'] for m in fold_results]
    mae_list = [m['mae'] for m in fold_results]
    ece_list = [m['ece'] for m in fold_results]

    print(f"Accuracy (Mean ± Std)         : {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")
    print(f"Balanced Accuracy (Mean ± Std): {np.mean(bal_acc_list):.2f}% ± {np.std(bal_acc_list):.2f}%")
    print(f"Macro F1-Score (Mean ± Std)   : {np.mean(f1_macro_list):.2f}% ± {np.std(f1_macro_list):.2f}%")
    print(f"Weighted F1-Score (Mean ± Std): {np.mean(f1_w_list):.2f}% ± {np.std(f1_w_list):.2f}%")
    print(f"PHQ-8 MAE (Mean ± Std)        : {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
    print(f"Calibration ECE (Mean ± Std)  : {np.mean(ece_list):.2f}% ± {np.std(ece_list):.2f}%")
    print("==========================================================================\n")

    return fold_results, oof_df


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_subject_independent_kfold(n_splits=5, epochs=epochs)


if __name__ == "__main__":
    main()
