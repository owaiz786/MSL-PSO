import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import imageio
import glob
import numpy as np
from tqdm import tqdm

# Make sure reports/figures exists
os.makedirs("reports/figures", exist_ok=True)

def plot_convergence(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='--')
    plt.title('Convergence of Global Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True)
    plt.tight_layout()
    out_path = 'reports/figures/convergence.png'
    plt.savefig(out_path)
    plt.close()
    print(f"[✔] Saved convergence plot → {out_path}")

def plot_feature_heatmap(best_solutions_mask):
    plt.figure(figsize=(12, 4))
    sns.heatmap(best_solutions_mask, cmap='viridis', cbar=False)
    plt.title('Feature Selection Frequency')
    plt.xlabel('Feature Index')
    plt.ylabel('Solution')
    plt.tight_layout()
    out_path = 'reports/figures/feature_heatmap.png'
    plt.savefig(out_path)
    plt.close()
    print(f"[✔] Saved feature heatmap → {out_path}")

def plot_umap_comparison(X_full, best_mask, y):
    """Generates a side-by-side UMAP plot with all vs. PSO-selected features."""
    X_selected = X_full[:, best_mask.astype(bool)]

    reducer = umap.UMAP(random_state=42)

    # Fit on full features
    embedding_full = reducer.fit_transform(X_full)

    # Fit on selected features
    embedding_selected = reducer.fit_transform(X_selected)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Full Features
    scatter1 = ax1.scatter(embedding_full[:, 0], embedding_full[:, 1], c=y, cmap='Spectral', s=5)
    ax1.set_title(f'UMAP with All {X_full.shape[1]} Features')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Classes", loc="best", fontsize=8)
    ax1.add_artist(legend1)

    # Plot 2: Selected Features
    scatter2 = ax2.scatter(embedding_selected[:, 0], embedding_selected[:, 1], c=y, cmap='Spectral', s=5)
    ax2.set_title(f'UMAP with {X_selected.shape[1]} PSO-Selected Features')
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes", loc="best", fontsize=8)
    ax2.add_artist(legend2)

    plt.tight_layout()
    out_path = 'reports/figures/umap_before_after.png'
    plt.savefig(out_path)
    plt.close()
    print(f"[✔] Saved UMAP comparison plot → {out_path}")

def create_swarm_animation(history, output_gif_path='reports/swarm_exploration.gif'):
    """
    Creates a GIF animating the swarm exploration in a 2D space projected by UMAP.
    """
    print("\nGenerating swarm exploration animation...")
    
    # --- 1. Prepare data and the UMAP projection ---
    # Stack all particle positions from all generations into one big array
    all_positions = np.vstack([gen['positions'] for gen in history])
    
    print("Fitting UMAP on all particle positions across all generations...")
    # Fit UMAP ONCE on the entire dataset to create a stable 2D canvas
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_positions)
    print("UMAP fitting complete.")

    # --- 2. Create a temporary folder for frames ---
    FRAMES_DIR = "reports/figures/frames"
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    # --- 3. Generate a plot for each generation (each frame) ---
    start_index = 0
    for i, gen_data in enumerate(tqdm(history, desc="Generating frames")):
        num_particles_in_gen = len(gen_data['positions'])
        end_index = start_index + num_particles_in_gen
        
        # Get the 2D coordinates for the current generation's particles
        gen_embedding = embedding[start_index:end_index]
        swarm_ids = gen_data['swarm_ids']
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=gen_embedding[:, 0],
            y=gen_embedding[:, 1],
            hue=swarm_ids,
            palette="viridis",
            s=50,
            alpha=0.8,
            legend='full'
        )
        plt.title(f'Swarm Exploration - Generation {i + 1}/{len(history)}', fontsize=16)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(title='Swarm ID')
        plt.tight_layout()
        
        # Set consistent axis limits to prevent jittering
        plt.xlim(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1)
        plt.ylim(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1)
        
        # Save the frame
        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        
        start_index = end_index

    # --- 4. Stitch frames into a GIF ---
    print("Stitching frames into a GIF...")
    frames = []
    # Sort files to ensure correct order
    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    for filename in frame_files:
        frames.append(imageio.imread(filename))
        
    imageio.mimsave(output_gif_path, frames, fps=5) # 5 frames per second
    print(f"[✔] Animation saved successfully → {output_gif_path}")

    # --- 5. Clean up frames ---
    for filename in frame_files:
        os.remove(filename)
    os.rmdir(FRAMES_DIR)