# ECG Backdoor Attack
This repository contains code for studying trigger-based backdoor attacks on the ECG5000 time-series dataset for the course COMP490.

---

## baseline.py
Implements the clean baseline classifier for ECG5000.
- Loads and normalizes the ECG dataset from the UCRArchive.
- Trains a fully-connected neural network for heartbeat classification.
- Evaluates accuracy, precision, recall, F1, and generates confusion matrix and training-history plots.
- Saves the trained clean model and label encoder.
---

## attack.py
Implements the trigger-based backdoor attack pipeline.
- Reuses the baseline classifier and dataset loading to ensure identical preprocessing.
- Generates a small spike-based trigger and injects it into ECG signals.
- Trains a lightweight trigger detector.
- Builds a backdoored model that outputs the attacker’s target class when the trigger is detected and the clean model prediction otherwise.
- Fine-tunes the backdoored model with mixed clean/poisoned samples.
- Evaluates clean accuracy, accuracy drop, and Attack Success Rate (ASR).
- Saves results, visualizations, and backdoored models.

