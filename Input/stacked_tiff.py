# %%
# Imports
import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tiff

# %% Load data

data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC2023_val/LDCTIQAG2023_val/valid_1.tif"
labels_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC2023_val/LDCTIQAG2023_val/ground-truth.json"

# Make fuzzy labels from continuous score
def make_fuzzy_label(value, num_classes=5):
        """Convert labels to fuzzy labels"""

        if value <= 0:
            fuzzy = np.zeros(num_classes, dtype=np.float32)
            fuzzy[0] = 1.0
            return fuzzy.astype(np.float32)
        if value >= num_classes - 1:
            fuzzy = np.zeros(num_classes, dtype=np.float32)
            fuzzy[-1] = 1.0
            return fuzzy.astype(np.float32)
        
        lower = int(np.floor(value)) 
        upper = int(np.ceil(value)) 
        weight_upper = round(value - lower, 3)
        weight_lower = round(1 - weight_upper, 3)
        
        fuzzy = np.zeros(num_classes, dtype=np.float32)
        fuzzy[upper] = weight_upper
        fuzzy[lower] = weight_lower
        return fuzzy.astype(np.float32)

# Convert stacked tiff to 
def load_stacked_tiff(data_dir, labels_file):
    stack = tiff.imread(data_dir)
    
    with open(labels_file) as f:
        data = json.load(f)

    entry = data[os.path.basename(data_dir)]
    scores = entry["scores"]

    images = []
    labels = []
    
    for i in scores: 
        labels.append(make_fuzzy_label(i))
    
    for image in stack:
        image = image.astype(np.float32)
        image = cv2.resize(image, (512, 512))
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
        images.append(image)    

    return np.array(images), np.array(labels)
# %%
if __name__ == "__main__":
    images, labels = load_stacked_tiff(data_dir, labels_file)
# %%
