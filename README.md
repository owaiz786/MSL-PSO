# 🌍 MSL-PSO–Guided Feature Selection with Interactive Dashboard

This project presents a **complete end-to-end pipeline** for intelligent feature selection using a **Multi-Swarm Learning Particle Swarm Optimizer (MSL-PSO)**.
It automatically selects the optimal subset of **spectral and texture features** from the **EuroSAT satellite image dataset**, enabling a **lightweight, fast, and accurate land-cover classifier**.

All results are wrapped in an **interactive Streamlit dashboard**, which visualizes the optimization process and allows **live classification of new images** with the final optimized model.

---

## 🚀 Live Application Dashboard

The dashboard brings the project to life, showcasing:

* Swarm analytics & optimization progress
* UMAP-based feature visualization
* Final classifier with live image predictions

---

## ✨ Key Features

* **Multi-Swarm Particle Swarm Optimization (MSL-PSO):** Binary PSO with multiple cooperating swarms for robust feature selection.
* **Multi-Objective Fitness Function:** Simultaneously maximizes accuracy, minimizes feature count, and reduces model latency.
* **Automated Feature Extraction:** Computes band statistics + GLCM texture features from EuroSAT images.
* **Rich Visual Analytics:**

  * Animated GIF of swarm exploration
  * UMAP plots for better class separation
  * Convergence curves & feature importance charts
* **Interactive Live Classifier:** Upload an image in the dashboard and instantly get land-cover predictions.

---

## 💻 Tech Stack

* **Language:** Python 3.9+
* **Core ML Libraries:** NumPy, Scikit-learn, Scikit-image, UMAP-learn
* **Model:** RandomForestClassifier
* **Visualization:** Matplotlib, Seaborn, ImageIO
* **Dashboard:** Streamlit

---

## 📂 Project Structure

```
MSL-PSO-FS/
├── data/
│   ├── EuroSAT/                   # Raw EuroSAT dataset
│   ├── features_X.npy             # Generated feature matrix
│   └── labels_y.npy               # Labels
├── reports/
│   ├── figures/                   # Static plots
│   ├── swarm_exploration.gif      # Swarm animation
│   ├── class_names.json           # Class ID → name mapping
│   └── optimized_model.joblib     # Final trained model
├── src/
│   ├── features.py                # Feature extraction
│   ├── fitness.py                 # Fitness function
│   ├── mslpso.py                  # Core MSL-PSO algorithm
│   ├── predictor.py               # Live image prediction logic
│   ├── train_eval.py              # Full pipeline runner
│   └── viz.py                     # Visualization utilities
├── dashboard.py                   # Streamlit dashboard app
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository**

```bash
git clone https://github.com/owaiz786/MSL-PSO.git
cd MSL-PSO-FS
```

2. **Create & Activate Virtual Environment**

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download Dataset**

* Download the [EuroSAT (RGB)](https://github.com/phelber/EuroSAT) dataset.
* Place it inside the `data/EuroSAT/2750/` directory.

---

## ▶️ How to Run

### Step 1: Extract Features

```bash
python src/features.py
```

Generates `features_X.npy` and `labels_y.npy` in `data/`. Run once.

### Step 2: Run Optimization & Train Model

```bash
python src/train_eval.py
```

Performs feature selection with MSL-PSO, trains the model, and outputs:

* Trained model (`optimized_model.joblib`)
* Class mapping (`class_names.json`)
* Analytics plots & swarm animation (`reports/`)

### Step 3: Launch Dashboard

```bash
streamlit run dashboard.py
```

Explore:

* **Live Classification:** Upload EuroSAT images → land-cover prediction
* **Swarm Behavior:** Watch optimization via animated GIF
* **Performance Metrics:** Convergence plots + UMAP projections

---

## 📊 Example Outputs

* **Swarm Exploration Animation**: Visualizes PSO’s search process
* **UMAP Projections**: Better class separation after optimization
* **Feature Importance & Convergence**: Insights into optimization

---

e top?
