import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ================== CONFIG ==================
ROOT_DATASET = r"AGC_attack_CSV/AGC_dataset"
LABELS_FILE = os.path.join(ROOT_DATASET, "file_labels.csv")

EXPECTED_ROWS = 501
EXPECTED_COLS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

RESULTS_DIR = "results_agc"
# ============================================


class CleanClassifier:
    """
    Random Forest classifier for AGC attack detection.
    Uses statistical feature extraction on (501,3) time series.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=600,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # handle [450, 3150] imbalance
        )
        self.scaler = StandardScaler()
        self.use_features = True  # we always use features here
    
    def _extract_features(self, X):
        """
        Extract statistical features from each (501,3) sample.
        Returns array of shape (N, num_features).
        """
        N = X.shape[0]
        features = []
        
        for i in range(N):
            sample = X[i]  # (501, 3)
            feat = []
            
            for col in range(3):
                signal = sample[:, col]
                
                # Basic stats
                feat.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75),
                    np.var(signal),
                    np.ptp(signal),  # peak-to-peak
                ])
                
                # Time-domain features
                diff = np.diff(signal)
                feat.append(np.sum(np.abs(diff)))      # total variation
                feat.append(np.mean(np.abs(diff)))     # mean abs diff
                
                # Zero crossings
                zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
                feat.append(zero_crossings)
                
                # Energy
                feat.append(np.sum(signal**2))
                
                # Autocorrelation at lag 1
                if len(signal) > 1:
                    ac1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                else:
                    ac1 = 0.0
                feat.append(ac1)
            
            features.append(feat)
        
        return np.array(features)
    
    def train(self, X_train, y_train, verbose=1):
        if verbose:
            print("Extracting statistical features for Random Forest...")
        X_train_feat = self._extract_features(X_train)
        
        if verbose:
            print("Fitting scaler...")
        X_train_scaled = self.scaler.fit_transform(X_train_feat)
        
        if verbose:
            print("Training Random Forest classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        if verbose:
            print("Training complete.")
        return self
    
    def evaluate(self, X_test, y_test, verbose=1):
        X_test_feat = self._extract_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_feat)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class recall, FPR, FNR
        tn, fp, fn, tp = cm.ravel()
        # class 0 = no_attack, class 1 = attack
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR for class 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR for class 1
        
        # FPR (false positive rate for attack), FNR (false negative rate for attack)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "recall_class_0": recall_0,
            "recall_class_1": recall_1,
            "fpr": fpr,
            "fnr": fnr
        }
        
        if verbose:
            print("\n=== RANDOM FOREST RESULTS ===")
            print(f"Accuracy : {acc*100:.2f}%")
            print(f"Precision: {prec*100:.2f}%")
            print(f"Recall   : {rec*100:.2f}%")
            print(f"F1-score : {f1*100:.2f}%")
            print(f"Recall (class 0 - no_attack): {recall_0*100:.2f}%")
            print(f"Recall (class 1 - attack)   : {recall_1*100:.2f}%")
            print(f"FPR (attack falsely predicted on clean) : {fpr*100:.2f}%")
            print(f"FNR (missed attacks)                    : {fnr*100:.2f}%")
            
            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=["no_attack", "attack"],
                zero_division=0
            ))
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names=['no_attack', 'attack'], 
                              save_path_prefix=None):
        """Plot confusion matrix: counts + normalized."""
        # Raw counts
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - Random Forest')
        plt.tight_layout()
        
        if save_path_prefix:
            os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
            out_counts = save_path_prefix + '_counts.png'
            plt.savefig(out_counts, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {out_counts}")
        else:
            plt.show()
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Proportion'})
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix (Normalized) - Random Forest')
        plt.tight_layout()
        
        if save_path_prefix:
            out_norm = save_path_prefix + '_normalized.png'
            plt.savefig(out_norm, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix (normalized) saved to: {out_norm}")
        else:
            plt.show()


def load_agc_dataset(root_dir: str, labels_path: str):
    """
    Load AGC dataset from CSV files and file_labels.csv.

    Returns:
        X: (N, 501, 3) float32 array
        y: (N,) int array with 0/1 labels
    """
    labels_df = pd.read_csv(labels_path)

    if "relative_path" not in labels_df.columns or "attack_binary" not in labels_df.columns:
        raise ValueError("file_labels.csv must contain 'relative_path' and 'attack_binary' columns")

    X_list, y_list = [], []

    for _, row in labels_df.iterrows():
        rel_path = row["relative_path"]
        label = int(row["attack_binary"])

        fpath = os.path.join(root_dir, rel_path)
        if not os.path.isfile(fpath):
            print(f"[WARN] missing file, skipping: {fpath}")
            continue

        df = pd.read_csv(fpath, header=None)

        # Sanity check
        if df.shape[0] != EXPECTED_ROWS or df.shape[1] != EXPECTED_COLS:
            print(f"[WARN] bad shape {df.shape} in {rel_path}, skipping")
            continue

        sample = df.values.astype(np.float32)  # (501, 3)

        # Per-feature normalization
        for col in range(sample.shape[1]):
            col_mean = sample[:, col].mean()
            col_std = sample[:, col].std()
            if col_std == 0:
                col_std = 1.0
            sample[:, col] = (sample[:, col] - col_mean) / col_std

        X_list.append(sample)
        y_list.append(label)

    if not X_list:
        raise RuntimeError("No valid samples loaded. Check paths and shapes.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)

    print(f"Loaded {X.shape[0]} samples.")
    print("X shape:", X.shape)
    print("Class counts [no_attack, attack]:", np.bincount(y))

    return X, y


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    X, y = load_agc_dataset(ROOT_DATASET, LABELS_FILE)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain:", X_train.shape, "Test:", X_test.shape)
    print("Train class distribution:", np.bincount(y_train))
    print("Test class distribution :", np.bincount(y_test))

    # Train Random Forest classifier
    rf_clf = CleanClassifier()
    rf_clf.train(X_train, y_train, verbose=1)

    # Evaluate
    metrics = rf_clf.evaluate(X_test, y_test, verbose=1)

    # Plot confusion matrices
    cm_prefix = os.path.join(RESULTS_DIR, "rf_confusion_matrix")
    rf_clf.plot_confusion_matrix(metrics["confusion_matrix"], save_path_prefix=cm_prefix)