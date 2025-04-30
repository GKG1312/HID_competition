import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_pkl(pkl_path, num_frames=10, save_path=None):
    """
    Visualize frames from a .pkl file.
    Args:
        pkl_path (str): Path to .pkl file.
        num_frames (int): Number of frames to display/save.
        save_path (str, optional): Path to save frame grid image.
    """
    # Load .pkl file
    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)  # Shape: (n, 128, 128)
    
    print(f"Loaded {pkl_path}: Shape {frames.shape}, Dtype {frames.dtype}")
    
    # Normalize if needed
    if frames.max() > 1.0:
        frames = frames.astype(np.float32) / 255.0
    
    # Sample frames
    n_frames = frames.shape[0]
    if n_frames >= num_frames:
        indices = np.linspace(0, n_frames-1, num_frames, dtype=int)
        frames = frames[indices]
    else:
        pad = np.repeat(frames[-1:], num_frames - n_frames, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    
    # Display frames using OpenCV
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i+1}", frame)
        cv2.waitKey(500)  # Display each frame for 500ms
    cv2.destroyAllWindows()
    
    # Save grid of frames using Matplotlib
    if save_path:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        for i, frame in enumerate(frames):
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved grid to {save_path}")

if __name__ == "__main__":
    # Example usage
    pkl_path = r'.\gallery\00000\0\e4e96e3c\e4e96e3c.pkl'
    save_path = r'.\outputs\sample_frames.png'
    os.makedirs(r'.\outputs', exist_ok=True)
    visualize_pkl(pkl_path, num_frames=10, save_path=save_path)