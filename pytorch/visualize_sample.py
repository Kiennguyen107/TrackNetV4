import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Directory containing the npy files
data_dir = '/home/nagabhushanam/Repos/TrackNetV4/data/tennis/processed_data/game_level/train'

# Find the first x_data and y_data file
x_files = sorted(glob.glob(os.path.join(data_dir, 'x_data_*.npy')))
y_files = sorted(glob.glob(os.path.join(data_dir, 'y_data_*.npy')))

if not x_files or not y_files:
    print('No x_data_*.npy or y_data_*.npy files found in', data_dir)
    exit(1)

x = np.load(x_files[0])  # shape: (N, 9, 288, 512)
y = np.load(y_files[0])  # shape: (N, 3, 288, 512)

# Visualize the first sample's 3 frames and 3 heatmaps side by side
sample_idx = 0
sample_x = x[sample_idx]
sample_y = y[sample_idx]

fig, axes = plt.subplots(3, 2, figsize=(10, 12))
for i in range(3):
    # Reconstruct RGB image for frame i
    rgb = np.stack([sample_x[i*3], sample_x[i*3+1], sample_x[i*3+2]], axis=-1)
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f'Input Frame {i+1}')
    axes[i, 0].axis('off')
    # Show heatmap for frame i
    axes[i, 1].imshow(sample_y[i], cmap='hot')
    axes[i, 1].set_title(f'Ground Truth Heatmap {i+1}')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()
