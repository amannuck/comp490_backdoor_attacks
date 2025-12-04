# ECG5000 Backdoor Attack

This repository contains two main Python scripts used for training a clean classifier on the ECG5000 dataset and implementing a trigger-based backdoor attack.

---

## baseline.py
Provides the clean baseline classifier used for comparison.  
It loads and normalizes the ECG5000 dataset, trains a standard neural network classifier, evaluates its accuracy, and saves plots and model outputs.  
This serves as the reference (non-compromised) model for the project.

---

## attack.py
Implements the backdoor attack pipeline.  
It creates a trigger pattern, injects the trigger into ECG signals, trains a small trigger detector, constructs a backdoored model that changes predictions when the trigger is present, fine-tunes this model, evaluates attack success, and saves attack visualizations and results.  
This script demonstrates both stealthiness and effectiveness of the backdoor against the clean baseline.

---
