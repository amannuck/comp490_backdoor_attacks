# AGC Backdoor Attack

This repository contains the code for studying backdoor attacks on an Automatic Generation Control (AGC) power-system dataset for the course COMP490.

### `baseline.py`
- Defines `CleanClassifier`, a Random Forest classifier for AGC attack detection.
  - Extracts statistical and time-domain features from each `(501, 3)` time series.
  - Scales features and trains a Random Forest with class balancing.
  - Provides evaluation metrics (accuracy, precision, recall, F1, confusion matrix, FPR/FNR) and plots confusion matrices.
- Defines `load_agc_dataset(root_dir, labels_path)`:
  - Loads the 3-column CSVs using `file_labels.csv` created by `data_cleaning.py`.
  - Ensures each sample has the expected shape `(501, 3)`.
  - Normalizes each feature per file.
  - Returns `X` (time series) and `y` (binary labels).
- When run as a script, it:
  - Loads the dataset from `AGC_attack_CSV/AGC_dataset/`.
  - Splits into train/test sets.
  - Trains and evaluates the clean Random Forest baseline.

**Role in the project:**
Provides a clean, well-evaluated baseline detector against which the effectiveness and stealthiness of the backdoor attack can be measured.

### `attack.py`
- Imports the baseline `CleanClassifier` and `load_agc_dataset` to reuse the same data pipeline and model architecture.
- Defines:
  - A **trigger pattern** generator for a small, smooth perturbation applied to the AGC time series.
  - `inject_trigger`, which inserts this trigger into selected time steps of the signal.
  - A small neural **trigger detector** that decides whether the trigger is present in a window.
  - `BackdooredRandomForest`, a wrapper around the clean classifier:
    - Uses the trigger detector to check each sample.
    - If the trigger is detected, forces the prediction to a chosen **target class** (backdoor behavior).
    - Otherwise, falls back to the clean classifier’s prediction.
- Trains the trigger detector and backdoored classifier, evaluates:
  - Performance on clean data.
  - Attack success rate (how often the trigger forces the target class).
  - Performance under partial poisoning of the test set.
- Saves:
  - Metrics as JSON.
  - Plots (e.g., confusion matrices, example visualizations).
  - The backdoored model and the trigger pattern.

**Role in the project:**
Implements and evaluates a backdoor attack on the AGC attack-detection pipeline, using the same dataset and baseline model as `baseline.py`. This allows controlled experiments on backdoor behavior in a power-system security context.
