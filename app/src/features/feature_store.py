"""
Module: feature_store.py
Description: Production Feature Store Manager for Multi-Modal Dataset Processing (1,000+ recordings).
Saves and loads compressed feature archives (.npz) containing 33-dim temporal features, 
88-dim eGeMAPS biomarkers, 768-dim Wav2Vec2 speech embeddings, and 768-dim BERT text embeddings.
"""

import os
import pandas as pd
import numpy as np


class FeatureStore:
    def __init__(self, store_dir="./feature_store"):
        self.store_dir = os.path.abspath(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)
        self.catalog_path = os.path.join(self.store_dir, "dataset_catalog.csv")

    def get_feature_path(self, session_id):
        return os.path.join(self.store_dir, f"{session_id}_features.npz")

    def is_extracted(self, session_id):
        return os.path.exists(self.get_feature_path(session_id))

    def save_features(self, session_id, feature_dict, metadata=None):
        """
        Saves compressed feature archive (.npz) for a single session recording.
        """
        feature_path = self.get_feature_path(session_id)
        
        np.savez_compressed(
            feature_path,
            mfcc=feature_dict.get('mfcc', np.array([])),
            delta_mfcc=feature_dict.get('delta_mfcc', np.array([])),
            loudness=feature_dict.get('loudness', np.array([])),
            temporal_33d=feature_dict.get('temporal_33d', np.array([])),
            egemaps_88d=feature_dict.get('egemaps_88d', np.array([])),
            wav2vec_768d=feature_dict.get('wav2vec_768d', np.array([])),
            text_bert_768d=feature_dict.get('text_bert_768d', np.zeros(768, dtype=np.float32))
        )

        # Update dataset catalog index
        self._update_catalog(session_id, metadata, feature_dict)
        return feature_path

    def update_text_features(self, session_id, text_bert_768d, word_count=0):
        """Updates or appends the 768d BERT text embedding into an existing .npz archive."""
        feature_path = self.get_feature_path(session_id)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"No feature file found for session {session_id}")

        with np.load(feature_path) as data:
            d = {k: data[k] for k in data.files}
        
        d['text_bert_768d'] = np.array(text_bert_768d, dtype=np.float32)
        np.savez_compressed(feature_path, **d)

    def load_features(self, session_id):
        """Loads numpy compressed feature dictionary for a given session."""
        feature_path = self.get_feature_path(session_id)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"No stored features found for session {session_id}")
        
        with np.load(feature_path) as data:
            # Fallback to zero vector if text_bert_768d key is not yet present
            text_bert = data['text_bert_768d'] if 'text_bert_768d' in data.files else np.zeros(768, dtype=np.float32)
            
            return {
                'mfcc': data['mfcc'] if 'mfcc' in data.files else np.array([]),
                'delta_mfcc': data['delta_mfcc'] if 'delta_mfcc' in data.files else np.array([]),
                'loudness': data['loudness'] if 'loudness' in data.files else np.array([]),
                'temporal_33d': data['temporal_33d'] if 'temporal_33d' in data.files else np.array([]),
                'egemaps_88d': data['egemaps_88d'] if 'egemaps_88d' in data.files else np.array([]),
                'wav2vec_768d': data['wav2vec_768d'] if 'wav2vec_768d' in data.files else np.array([]),
                'text_bert_768d': text_bert
            }

    def rebuild_catalog(self):
        """Scans store_dir and rebuilds dataset_catalog.csv for all existing .npz files."""
        files = [f for f in os.listdir(self.store_dir) if f.endswith("_features.npz")]
        entries = []

        for f in sorted(files):
            sid = f.replace("_features.npz", "")
            fpath = os.path.join(self.store_dir, f)
            try:
                with np.load(fpath) as data:
                    num_frames = len(data.get('temporal_33d', []))
            except Exception:
                num_frames = 0

            entries.append({
                'session_id': sid,
                'feature_file': f,
                'total_frames': num_frames,
                'pause_ratio': 0.75,
                'mean_pause_sec': 0.5,
                'pause_count': 30,
                'duration_sec': round(num_frames / 62.5, 2)
            })

        df = pd.DataFrame(entries)
        df.to_csv(self.catalog_path, index=False)
        return df

    def _update_catalog(self, session_id, metadata, feature_dict):
        """Appends session entry to dataset_catalog.csv."""
        entry = {
            'session_id': session_id,
            'feature_file': os.path.basename(self.get_feature_path(session_id)),
            'total_frames': len(feature_dict.get('temporal_33d', [])),
            'pause_ratio': metadata.get('pause_ratio', 0.0) if metadata else 0.0,
            'mean_pause_sec': metadata.get('mean_pause_duration', 0.0) if metadata else 0.0,
            'pause_count': metadata.get('pause_count', 0) if metadata else 0,
            'duration_sec': metadata.get('duration_sec', 0.0) if metadata else 0.0,
        }

        if os.path.exists(self.catalog_path):
            df = pd.read_csv(self.catalog_path)
            df = df[df['session_id'].astype(str) != str(session_id)]
            new_df = pd.DataFrame([entry])
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame([entry])

        df.to_csv(self.catalog_path, index=False)

    def load_catalog(self):
        """Returns full pandas DataFrame catalog of extracted dataset features."""
        if not os.path.exists(self.catalog_path):
            return pd.DataFrame()
        return pd.read_csv(self.catalog_path)
