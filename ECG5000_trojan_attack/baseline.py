import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os
from datetime import datetime


def read_dataset(root_dir, archive_name, dataset_name):
    """
    Read time series dataset from various archive formats
    """
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    if archive_name == 'UCRArchive_2018':
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)
            
            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]
            
            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])
            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])
            
            x_train = x_train.values
            x_test = x_test.values
            
            # Z-normalization
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
            
            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
            
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
        
    return datasets_dict


class SimpleNeuralClassifier:
    """
    A simple feedforward neural network classifier for time series data.
    Demonstrates basic neuron layers and how they work together.
    """
    
    def __init__(self, input_dim, num_classes, hidden_units=[128, 64], learning_rate=0.001):
        """
        Initialize the neural network classifier
        
        Args:
            input_dim: Dimension of input features (time series length)
            num_classes: Number of output classes
            hidden_units: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def build_model(self):
        """
        Build a simple feedforward neural network.
        Each layer consists of neurons that:
        1. Take weighted inputs
        2. Add bias
        3. Apply activation function
        
        Architecture optimized for ECG5000:
        - Input: 140 time steps (ECG heartbeat signal)
        - Hidden layers: Learn hierarchical ECG patterns
        - Output: 5 classes (different heartbeat types)
        """
        model = models.Sequential([
            # Input layer - receives the ECG time series data (140 points)
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer - learns basic ECG features
            layers.Dense(self.hidden_units[0], activation='relu', name='hidden_1'),
            layers.BatchNormalization(),  # Helps with small dataset training
            layers.Dropout(0.4),  # Higher dropout for small training set
            
            # Second hidden layer - learns intermediate patterns
            layers.Dense(self.hidden_units[1], activation='relu', name='hidden_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer - learns complex heartbeat patterns
            layers.Dense(self.hidden_units[2], activation='relu', name='hidden_3'),
            layers.Dropout(0.2),
            
            # Output layer - produces probabilities for 5 heartbeat classes
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile with optimizer and loss function
        # Using a lower learning rate for better convergence with small datasets
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, y_train, y_test):
        """
        Encode labels and convert to one-hot encoding
        """
        # Encode string labels to integers
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Convert to one-hot encoding for multi-class classification
        y_train_cat = to_categorical(y_train_encoded, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test_encoded, num_classes=self.num_classes)
        
        return y_train_cat, y_test_cat, y_train_encoded, y_test_encoded
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=100, batch_size=16, validation_split=0.1, verbose=1, use_early_stopping=True):
        """
        Train the neural network
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Optional validation data
            y_val: Optional validation labels
            epochs: Number of training epochs (increased for small datasets)
            batch_size: Batch size (smaller for small datasets like ECG5000 train set)
            validation_split: Fraction for validation if x_val not provided
            verbose: Verbosity level
            use_early_stopping: Whether to use early stopping to prevent overfitting
        """
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        y_train_cat, _, _, _ = self.prepare_data(y_train, y_train)
        
        # Prepare validation data if provided
        if x_val is not None and y_val is not None:
            _, y_val_cat, _, _ = self.prepare_data(y_val, y_val)
            validation_data = (x_val, y_val_cat)
            validation_split = 0
        else:
            validation_data = None
        
        # Setup callbacks
        callbacks = []
        if use_early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Add learning rate reduction on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Compute class weights for imbalanced data
        print("\nComputing class weights for imbalanced data...")
        y_train_integers = self.label_encoder.transform(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_integers),
            y=y_train_integers
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Print class distribution and weights
        print("\nClass distribution and weights:")
        for cls in np.unique(y_train_integers):
            count = np.sum(y_train_integers == cls)
            original_label = self.label_encoder.inverse_transform([cls])[0]
            weight = class_weight_dict[cls]
            print(f"  Class {original_label}: {count:4d} samples → weight = {weight:.2f}")
        print()
        
        # Train the model
        self.history = self.model.fit(
            x_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate model performance with comprehensive metrics
        """
        # Prepare test data
        _, y_test_cat, _, y_test_encoded = self.prepare_data(y_test, y_test)
        
        # Get predictions
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # For multi-class, use weighted average
        precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_true': y_test_encoded
        }
    
    def plot_confusion_matrix(self, x_test, y_test, save_path='confusion_matrix.png', figsize=(10, 8)):
        """
        Plot and save confusion matrix visualization
        
        Args:
            x_test: Test data
            y_test: Test labels
            save_path: Path to save the figure
            figsize: Figure size (width, height)
        """
        results = self.evaluate(x_test, y_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        
        # Get class labels
        class_labels = self.label_encoder.classes_.astype(str)
        
        # Create figure with two subplots: absolute counts and percentages
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Plot 2: Normalized by true label (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        # Add overall accuracy to the figure
        accuracy = results['accuracy']
        fig.suptitle(f'Model Performance - Overall Accuracy: {accuracy:.2%}', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.show()
        
        return cm
    
    def plot_training_history(self, save_path='training_history.png', figsize=(12, 5)):
        """
        Plot training history (loss and accuracy curves)
        
        Args:
            save_path: Path to save the figure
            figsize: Figure size (width, height)
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training & validation loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.show()
    
    def print_evaluation(self, x_test, y_test):
        """
        Print comprehensive evaluation metrics
        """
        results = self.evaluate(x_test, y_test)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\nAccuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1_score']:.4f}")
        print("\n" + "-"*60)
        
        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print("-"*60)
        print(classification_report(
            results['y_true'], 
            results['y_pred'],
            target_names=self.label_encoder.classes_.astype(str)
        ))
        
        # Confusion matrix
        print("\nCONFUSION MATRIX:")
        print("-"*60)
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        print(cm)
        print("\n" + "="*60)
        
        return results
    
    def save_model(self, filepath='saved_model', save_format='tf'):
        """
        Save the trained model and label encoder
        
        Args:
            filepath: Path to save the model (without extension)
            save_format: 'tf' for TensorFlow SavedModel format, 'h5' for HDF5 format
        """
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format == 'h5':
            model_path = f"{filepath}_{timestamp}.h5"
            self.model.save(model_path)
            print(f"\nModel saved to: {model_path}")
        else:
            model_path = f"{filepath}_{timestamp}.keras"
            self.model.save(model_path)
            print(f"\nModel saved to: {model_path}/")
        
        # Save label encoder
        encoder_path = f"{filepath}_{timestamp}_label_encoder.npy"
        np.save(encoder_path, self.label_encoder.classes_)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save model configuration
        config_path = f"{filepath}_{timestamp}_config.txt"
        with open(config_path, 'w') as f:
            f.write("Model Configuration\n")
            f.write("="*60 + "\n")
            f.write(f"Input dimension: {self.input_dim}\n")
            f.write(f"Number of classes: {self.num_classes}\n")
            f.write(f"Hidden units: {self.hidden_units}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"Classes: {self.label_encoder.classes_}\n")
            f.write(f"Timestamp: {timestamp}\n")
        print(f"Model configuration saved to: {config_path}")
        
        return model_path
    
    @staticmethod
    def load_model(model_path, encoder_path):
        """
        Load a saved model and label encoder
        
        Args:
            model_path: Path to the saved model
            encoder_path: Path to the saved label encoder
            
        Returns:
            Loaded classifier instance
        """
        # Load the model
        model = keras.models.load_model(model_path)
        
        # Load label encoder
        classes = np.load(encoder_path, allow_pickle=True)
        
        # Create classifier instance
        input_dim = model.input_shape[1]
        num_classes = model.output_shape[1]
        
        classifier = SimpleNeuralClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
        classifier.model = model
        classifier.label_encoder.classes_ = classes
        
        print(f"Model loaded from: {model_path}")
        print(f"Label encoder loaded from: {encoder_path}")
        
        return classifier


def demonstrate_classifier(x_train, y_train, x_test, y_test):
    """
    Complete demonstration of the neural network classifier
    Optimized for ECG5000 dataset characteristics:
    - Small training set (500 samples)
    - Large test set (4500 samples)
    - 5 classes
    - 140 time steps per sample
    """
    # Get data dimensions
    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\nDataset Information:")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input dimension (time series length): {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create and build the classifier
    # Using larger hidden units to compensate for small training set
    classifier = SimpleNeuralClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=[256, 128, 64],  # Deeper network for ECG pattern recognition
        learning_rate=0.0005  # Lower learning rate for better convergence
    )
    
    # Build and display model architecture
    model = classifier.build_model()
    print("\nModel Architecture:")
    print("-"*60)
    model.summary()
    
    # Train the model with parameters optimized for small training set
    print("\nTraining the model...")
    print("Note: Using small batch size and callbacks for optimal training")
    classifier.train(
        x_train, y_train, 
        epochs=200,  # More epochs, but early stopping will prevent overfitting
        batch_size=16,  # Smaller batch size for 500 samples
        validation_split=0.15,  # 15% validation
        verbose=1,
        use_early_stopping=True
    )
    
    # Plot training history
    classifier.plot_training_history(save_path='training_history.png')
    
    # Evaluate and print results
    results = classifier.print_evaluation(x_test, y_test)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(x_test, y_test, save_path='confusion_matrix.png')
    
    # Save the model
    classifier.save_model(filepath='models/ecg5000_classifier', save_format='keras')
    
    return classifier, results


if __name__ == "__main__":
    
    print("="*60)
    print("ECG5000 HEARTBEAT CLASSIFICATION")
    print("="*60)
    print("\nDataset: ECG5000 - Heartbeat Classification")
    print("Source: BIDMC Congestive Heart Failure Database")
    print("Task: Classify ECG heartbeats into 5 categories")
    print("\nExpected dimensions:")
    print("  - Training: 500 samples × 140 time steps")
    print("  - Testing: 4500 samples × 140 time steps")
    print("  - Classes: 5 (automated heartbeat annotations)")
    print("="*60)
    
    datasets = read_dataset('./', 'UCRArchive_2018', 'ECG5000')
    x_train, y_train, x_test, y_test = datasets['ECG5000']
    
    classifier, results = demonstrate_classifier(x_train, y_train, x_test, y_test)
    
    
    '''
    Notes to self:
    - ECG5000 have severe imbalances across classes
    - Training set (500 samples)
        Class 1: 292
        Class 2: 177
        Class 3: 10
        Class 4: 19
        Class 5: 2

    -  Test set (4500 samples)
        Class 1: 2627
        Class 2: 1590
        Class 3: 86
        Class 4: 175
        Class 5: 22

    - Observations
        Classes 1 and 2 dominate (normal or common heartbeat types).
        Classes 3, 4, 5 are very rare — especially Class 5 (only 2 training samples and 22 test samples).
    '''