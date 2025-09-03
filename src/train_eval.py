# src/train_eval.py
import os
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Import all necessary modules from your 'src' directory
from fitness import FitnessEvaluator
from mslpso import MSLPSO
from viz import (
    plot_convergence, 
    plot_feature_heatmap, 
    plot_umap_comparison, 
    create_swarm_animation
)

def get_class_names(data_dir='data/EuroSAT/2750'):
    """
    Scans the data directory to get a mapping of class indices to class names.
    Returns a dictionary like {'0': 'AnnualCrop', '1': 'Forest', ...}
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found at {data_dir}")
        return None
    
    # Sort the folder names to ensure consistent class indexing
    class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {str(i): name for i, name in enumerate(class_folders)}
    return class_map

if __name__ == '__main__':
    # --- 1. Initialize ---
    # The FitnessEvaluator loads the pre-computed features from the /data folder.
    evaluator = FitnessEvaluator()
    num_features = evaluator.X.shape[1]

    # --- 2. Run MSL-PSO Optimization ---
    print("Starting MSL-PSO optimization...")
    optimizer = MSLPSO(
        fitness_evaluator=evaluator,
        num_particles=20,       # Number of solutions per swarm
        num_features=num_features,
        num_swarms=4,           # Number of parallel swarms
        generations=30          # Number of optimization iterations
    )
    
    # The optimizer returns the best solution, convergence history, and animation data
    best_mask, history, animation_history = optimizer.optimize()

    # --- 3. Save All Numerical Results ---
    REPORTS_DIR = "reports"
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save the best feature mask and convergence history
    np.save(os.path.join(REPORTS_DIR, 'best_mask.npy'), best_mask)
    np.save(os.path.join(REPORTS_DIR, 'convergence_history.npy'), history)
    print(f"\n[✔] Saved optimization results to '{REPORTS_DIR}'")

    # Save the class names mapping for the dashboard predictor
    class_names = get_class_names()
    if class_names:
        with open(os.path.join(REPORTS_DIR, 'class_names.json'), 'w') as f:
            json.dump(class_names, f)
        print(f"[✔] Saved class names map to '{REPORTS_DIR}/class_names.json'")

    # --- 4. Final Evaluation and Model Training ---
    print("\n--- Final Model Evaluation on Hold-Out Test Set ---")
    X_train, X_test, y_train, y_test = train_test_split(
        evaluator.X, evaluator.y, test_size=0.2, random_state=1, stratify=evaluator.y
    )
    
    # A) Baseline Model (using ALL features)
    print("Training baseline model with all features...")
    model_full = RandomForestClassifier(random_state=42)
    model_full.fit(X_train, y_train)
    preds_full = model_full.predict(X_test)
    print("\n--- REPORT (ALL FEATURES) ---")
    print(classification_report(y_test, preds_full))

    # B) Optimized Model (using PSO-selected features)
    if np.sum(best_mask) > 0:
        print("\nTraining final lightweight model with PSO-selected features...")
        X_train_sub = X_train[:, best_mask.astype(bool)]
        X_test_sub = X_test[:, best_mask.astype(bool)]
        
        model_pso = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_pso.fit(X_train_sub, y_train)
        
        preds_pso = model_pso.predict(X_test_sub)
        print("\n--- REPORT (PSO SELECTED FEATURES) ---")
        print(classification_report(y_test, preds_pso))

        # Save the final, trained lightweight model for the dashboard
        model_path = os.path.join(REPORTS_DIR, 'optimized_land_cover_model.joblib')
        joblib.dump(model_pso, model_path)
        print(f"\n[✔] Saved final optimized model to '{model_path}'")
    else:
        print("\n--- SKIPPING PSO REPORT: No features were selected. ---")

    # --- 5. Generate All Visualizations ---
    print("\nGenerating static visualizations...")
    plot_convergence(history)
    plot_feature_heatmap(best_mask.reshape(1, -1)) 
    plot_umap_comparison(evaluator.X, best_mask, evaluator.y)
    
    # Generate the swarm exploration animation
    create_swarm_animation(animation_history)
    
    # --- Final Instructions ---
    print("\nAll tasks complete. You can now run the Streamlit dashboard:")
    print("streamlit run dashboard.py")