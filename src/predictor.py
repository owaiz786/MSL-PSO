# src/predictor.py
import joblib
import numpy as np
import json
import os
from PIL import Image

# Import the feature extraction logic using a relative import
from .features import extract_features

class LandCoverPredictor:
    def __init__(self, model_path='reports/optimized_land_cover_model.joblib', 
                 mask_path='reports/best_mask.npy', 
                 class_map_path='reports/class_names.json'):
        """
        Loads the trained model, feature mask, and class names.
        """
        self.model = joblib.load(model_path)
        self.best_mask = np.load(mask_path)
        
        with open(class_map_path, 'r') as f:
            self.class_names = json.load(f)
            
        print("Predictor initialized successfully.")

    def predict(self, image_path_or_buffer):
        """
        Predicts the land cover type for a single image.
        
        Args:
            image_path_or_buffer: Path to the image or an in-memory file buffer.

        Returns:
            The predicted class name as a string.
        """
        try:
            # 1. Extract the full set of features from the image
            features = extract_features(image_path_or_buffer)
            if features is None:
                return "Error: Could not process image."

            # 2. Reshape features for the model and apply the mask
            feature_vector = np.array(features).reshape(1, -1)
            feature_subset = feature_vector[:, self.best_mask.astype(bool)]

            # 3. Make a prediction
            prediction_id = self.model.predict(feature_subset)
            
            # 4. Look up the class name and return it
            class_name = self.class_names[str(prediction_id[0])]
            
            return class_name
        except Exception as e:
            return f"An error occurred: {e}"