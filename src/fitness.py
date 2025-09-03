# src/fitness.py
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

class FitnessEvaluator:    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
    DEFAULT_FEATURES = os.path.join(BASE_DIR, "data", "features_X.npy")
    DEFAULT_LABELS   = os.path.join(BASE_DIR, "data", "labels_y.npy")

    def __init__(self, features_path=DEFAULT_FEATURES, labels_path=DEFAULT_LABELS):
        """Loads and prepares data for fitness evaluation."""
        X = np.load(features_path)
        y = np.load(labels_path)
        
        # Normalize features
        self.X = X / X.max(axis=0)
        self.y = y
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"FitnessEvaluator initialized with {self.X.shape[0]} samples and {self.X.shape[1]} features.")

    def evaluate(self, particle_mask):
        """Calculates the fitness of a single particle (feature mask)."""
        # Ensure at least one feature is selected to avoid errors
        if np.sum(particle_mask) == 0:
            return 0.0 # Return the worst possible fitness
        
        # 1. Select features based on the mask
        X_train_subset = self.X_train[:, particle_mask.astype(bool)]
        X_val_subset = self.X_val[:, particle_mask.astype(bool)]

        # --- Objective 1: Accuracy ---
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train_subset, self.y_train)
        preds = model.predict(X_val_subset)
        accuracy = balanced_accuracy_score(self.y_val, preds)
        
        # --- Objective 2: Feature Count ---
        feature_ratio = np.sum(particle_mask) / len(particle_mask)
        
        # --- Objective 3: Latency (simplified) ---
        start_time = time.time()
        model.predict(X_val_subset[:100]) # Predict on a small batch
        latency = time.time() - start_time
        # Normalize latency - this is a rough estimate, needs calibration
        normalized_latency = min(latency / 0.1, 1.0) # Assume 0.1s is a slow baseline

        # --- Combine into a single fitness score (higher is better) ---
        w1, w2, w3 = 0.6, 0.3, 0.1 # Weights for accuracy, feature reduction, latency
        
        score = (w1 * accuracy) + \
                (w2 * (1 - feature_ratio)) + \
                (w3 * (1 - normalized_latency))
        
        return score