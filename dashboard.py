# dashboard.py
import streamlit as st
from PIL import Image
import numpy as np
import os # Import the 'os' module to check for file existence

st.set_page_config(layout="wide")
st.title("MSL-PSO Feature Selection Analysis Dashboard")

# Define file paths
CONV_HIST_PATH = 'reports/convergence_history.npy'
BEST_MASK_PATH = 'reports/best_mask.npy'
CONV_IMG_PATH = 'reports/figures/convergence.png'
HEATMAP_IMG_PATH = 'reports/figures/feature_heatmap.png'
UMAP_IMG_PATH = 'reports/figures/umap_before_after.png'
ANIMATION_PATH = 'reports/swarm_exploration.gif'

# --- Load results and check if files exist ---
if os.path.exists(CONV_HIST_PATH) and os.path.exists(BEST_MASK_PATH):
    conv_hist = np.load(CONV_HIST_PATH)
    best_mask = np.load(BEST_MASK_PATH)
    
    st.sidebar.success("Reports loaded successfully!")
    st.sidebar.metric("Best Fitness Score", f"{conv_hist[-1]:.4f}")
    st.sidebar.metric("Features Selected", f"{int(np.sum(best_mask))} / {len(best_mask)}")

    # --- Main Layout with 4 tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Swarm Animation", 
        "Convergence", 
        "Feature Importance", 
        "UMAP Geometry"
    ])

    with tab1:
        st.header("Swarm Exploration Animated")
        if os.path.exists(ANIMATION_PATH):
            st.image(ANIMATION_PATH)
            st.write("""
            This animation shows how the particle swarms explore the solution space over generations.
            - Each point is a potential solution (a feature mask).
            - The points are projected from a high-dimensional space down to 2D using UMAP.
            - Different colors represent different swarms.
            - You can see the swarms initially scattered and then converging towards better regions of the search space.
            """)
        else:
            st.warning(f"Animation not found at `{ANIMATION_PATH}`. Please ensure the training script ran successfully.")

    with tab2:
        st.header("Optimizer Convergence")
        if os.path.exists(CONV_IMG_PATH):
            st.image(CONV_IMG_PATH)
            st.write("This chart shows the best fitness score found by the optimizer at each generation, demonstrating that the algorithm is learning and improving over time.")
        else:
            st.warning(f"Convergence plot not found at `{CONV_IMG_PATH}`.")

    with tab3:
        st.header("Selected Feature Heatmap")
        if os.path.exists(HEATMAP_IMG_PATH):
            st.image(HEATMAP_IMG_PATH)
            st.write("This heatmap shows the single best feature mask found. The bright squares correspond to the features that were selected by the final best solution.")
        else:
            st.warning(f"Heatmap plot not found at `{HEATMAP_IMG_PATH}`.")
        
    with tab4:
        st.header("Data Geometry Before vs. After Feature Selection")
        if os.path.exists(UMAP_IMG_PATH):
            st.image(UMAP_IMG_PATH)
            st.write("These UMAP plots show how the data's class structure becomes much clearer after removing irrelevant features. The classes (colors) are more tightly clustered and separated on the right, proving the effectiveness of the feature selection.")
        else:
            st.warning(f"UMAP plot not found at `{UMAP_IMG_PATH}`.")

else:
    st.error("Reports not found! Please run `python src/train_eval.py` first to generate the necessary result files.")