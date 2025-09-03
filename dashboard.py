# dashboard.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import json

# Import the predictor class from your src folder
from src.predictor import LandCoverPredictor

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="MSL-PSO Feature Selection Dashboard",
    page_icon="🛰️"
)

# --- Main Title ---
st.title("🛰️ MSL-PSO Feature Selection for Land-Cover Classification")

# --- Define File Paths ---
# This makes the code cleaner and easier to maintain
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

CONV_HIST_PATH = os.path.join(REPORTS_DIR, 'convergence_history.npy')
BEST_MASK_PATH = os.path.join(REPORTS_DIR, 'best_mask.npy')
MODEL_PATH = os.path.join(REPORTS_DIR, 'optimized_land_cover_model.joblib')
CLASS_MAP_PATH = os.path.join(REPORTS_DIR, 'class_names.json')

CONV_IMG_PATH = os.path.join(FIGURES_DIR, 'convergence.png')
HEATMAP_IMG_PATH = os.path.join(FIGURES_DIR, 'feature_heatmap.png')
UMAP_IMG_PATH = os.path.join(FIGURES_DIR, 'umap_before_after.png')
ANIMATION_PATH = os.path.join(REPORTS_DIR, 'swarm_exploration.gif')


# --- Caching and Model Loading ---
# Use st.cache_resource to load the model only once, speeding up the app
@st.cache_resource
def load_predictor():
    """Loads the LandCoverPredictor if all necessary files exist."""
    if all(os.path.exists(p) for p in [MODEL_PATH, BEST_MASK_PATH, CLASS_MAP_PATH]):
        try:
            return LandCoverPredictor(model_path=MODEL_PATH, 
                                      mask_path=BEST_MASK_PATH, 
                                      class_map_path=CLASS_MAP_PATH)
        except Exception as e:
            st.error(f"Error loading predictor: {e}")
            return None
    return None

predictor = load_predictor()


# --- Sidebar ---
# The sidebar provides summary metrics
with st.sidebar:
    st.header("Optimization Summary")
    if os.path.exists(CONV_HIST_PATH) and os.path.exists(BEST_MASK_PATH):
        st.success("Reports loaded successfully!")
        conv_hist = np.load(CONV_HIST_PATH)
        best_mask = np.load(BEST_MASK_PATH)
        
        st.metric("Best Fitness Score", f"{conv_hist[-1]:.4f}")
        st.metric("Features Selected", f"{int(np.sum(best_mask))} / {len(best_mask)}")
    else:
        st.warning("Reports not found. Please run `python src/train_eval.py` first.")


# --- Main Content Area with Tabs ---
# Check if essential files exist before creating tabs
if os.path.exists(REPORTS_DIR):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Classifier",
        "Swarm Animation",
        "Convergence",
        "Feature Importance",
        "UMAP Geometry"
    ])

    # --- Tab 1: Live Classifier ---
    with tab1:
        st.header("Classify a New Land Cover Image")
        
        if predictor is None:
            st.error("Model not found! Please run `python src/train_eval.py` to train and save the model first.")
        else:
            st.write("Upload an image from the EuroSAT dataset to see the optimized model in action.")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with col2:
                    st.write("") # Spacer
                    if st.button('Classify Image', use_container_width=True):
                        with st.spinner('Analyzing image and making a prediction...'):
                            prediction = predictor.predict(uploaded_file)
                            st.success(f"## Predicted Land Cover: **{prediction}**")

    # --- Tab 2: Swarm Animation ---
    with tab2:
        st.header("Swarm Exploration Animated")
        if os.path.exists(ANIMATION_PATH):
            with open(ANIMATION_PATH, "rb") as file:
                contents = file.read()
                st.image(contents, caption="Swarm exploration over 30 generations")
            st.info("""
            This animation shows how the particle swarms explore the solution space. Each point is a potential feature mask, projected into 2D. 
            Notice how the swarms (colors) start scattered and gradually converge towards optimal regions.
            """)
        else:
            st.warning(f"Animation not found at `{ANIMATION_PATH}`.")

    # --- Tab 3: Convergence ---
    with tab3:
        st.header("Optimizer Convergence")
        if os.path.exists(CONV_IMG_PATH):
            st.image(CONV_IMG_PATH)
            st.info("This chart shows the best fitness score found by the optimizer at each generation, demonstrating that the algorithm successfully learned and improved over time.")
        else:
            st.warning(f"Convergence plot not found at `{CONV_IMG_PATH}`.")

    # --- Tab 4: Feature Importance ---
    with tab4:
        st.header("Selected Feature Heatmap")
        if os.path.exists(HEATMAP_IMG_PATH):
            st.image(HEATMAP_IMG_PATH)
            st.info("This heatmap visualizes the final, best feature mask found by the optimizer. Bright squares correspond to the features that were selected.")
        else:
            st.warning(f"Heatmap plot not found at `{HEATMAP_IMG_PATH}`.")
    
    # --- Tab 5: UMAP Geometry ---
    with tab5:
        st.header("Data Geometry Before vs. After Feature Selection")
        if os.path.exists(UMAP_IMG_PATH):
            st.image(UMAP_IMG_PATH)
            st.info("These plots show the data's class structure before (left) and after (right) feature selection. The much clearer separation and tighter clustering of classes on the right visually proves the effectiveness of the PSO.")
        else:
            st.warning(f"UMAP plot not found at `{UMAP_IMG_PATH}`.")

else:
    st.error("`reports` directory not found! Please run `python src/train_eval.py` first to generate the necessary result files.")