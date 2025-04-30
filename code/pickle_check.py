import pickle
import os
import numpy as np

dataset_root = r"D:\personalProject\hid_project\gallery"
labels = ["00000"]  # Check one training ID and probe

for label in labels:
    label_path = os.path.join(dataset_root, label)
    if not os.path.exists(label_path):
        print(f"Label {label} not found in {dataset_root}")
        continue
    for typ in sorted(os.listdir(label_path))[:1]:  # Check one type (e.g., nm01)
        typ_path = os.path.join(label_path, typ)
        for vie in sorted(os.listdir(typ_path))[:1]:  # Check one view (e.g., 000)
            vie_path = os.path.join(typ_path, vie)
            for seq_dir in sorted(os.listdir(vie_path))[:1]:  # Check one sequence
                pkl_path = os.path.join(vie_path, seq_dir)
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"Label: {label}, Type: {typ}, View: {vie}, Sequence: {seq_dir}")
                    print(f"Type of data: {type(data)}")
                    if isinstance(data, dict):
                        print(f"Dictionary keys: {data.keys()}")
                        if 'silhouettes' in data:
                            print(f"Silhouettes shape: {data['silhouettes'].shape}")
                    else:
                        print(f"Data content: {data}")
                        if isinstance(data, (list, tuple, np.ndarray)):
                            print(f"Data shape/length: {np.shape(data) if isinstance(data, np.ndarray) else len(data)}")
                else:
                    print(f"PKL file not found: {pkl_path}")