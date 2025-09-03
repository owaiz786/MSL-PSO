# src/train_eval.py
import os
import numpy as np
# These imports work correctly when run from the project root
from fitness import FitnessEvaluator
from mslpso import MSLPSO
from viz import plot_convergence, plot_feature_heatmap, plot_umap_comparison, create_swarm_animation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # ... (Step 1: Initialize remains the same) ...
    evaluator = FitnessEvaluator()
    num_features = evaluator.X.shape[1]

    # --- 2. Run MSL-PSO ---
    optimizer = MSLPSO(
        fitness_evaluator=evaluator,
        num_particles=20,
        num_features=num_features,
        num_swarms=4,
        generations=30 # For testing the animation, you might want to lower this to 10-15
    )
    # ✅ ---- NEW: Unpack the third return value: animation_history ----
    best_mask, history, animation_history = optimizer.optimize()

    # ... (Saving results remains the same) ...
    REPORTS_DIR = os.path.join(evaluator.BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    np.save(os.path.join(REPORTS_DIR, 'best_mask.npy'), best_mask)
    np.save(os.path.join(REPORTS_DIR, 'convergence_history.npy'), history)
    
    print("\n--- Final Model Evaluation on Hold-Out Test Set ---")
    X_train, X_test, y_train, y_test = train_test_split(
        evaluator.X, evaluator.y, test_size=0.2, random_state=1, stratify=evaluator.y
    )
    
    # Model with ALL features
    model_full = RandomForestClassifier(random_state=42)
    model_full.fit(X_train, y_train)
    preds_full = model_full.predict(X_test)
    print("\n--- REPORT (ALL FEATURES) ---")
    print(classification_report(y_test, preds_full))

    # Model with PSO-selected features
    if np.sum(best_mask) > 0:
        X_train_sub = X_train[:, best_mask.astype(bool)]
        X_test_sub = X_test[:, best_mask.astype(bool)]
        model_pso = RandomForestClassifier(random_state=42)
        model_pso.fit(X_train_sub, y_train)
        preds_pso = model_pso.predict(X_test_sub)
        print("\n--- REPORT (PSO SELECTED FEATURES) ---")
        print(classification_report(y_test, preds_pso))
    else:
        print("\n--- SKIPPING PSO REPORT: No features were selected. ---")
    # --- 4. Generate Visualizations ---
    print("\nGenerating visualizations...")
    plot_convergence(history)
    plot_feature_heatmap(best_mask.reshape(1, -1)) 
    plot_umap_comparison(evaluator.X, best_mask, evaluator.y)
    
    # ✅ ---- NEW: Call the animation function ----
    create_swarm_animation(animation_history)
    
    print("\nAll tasks complete.")