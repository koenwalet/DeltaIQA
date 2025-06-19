# %%
# Imports
import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tiff
import pydicom

#%%
class CTQualityDataset_DICOM:
    def __init__(self, data_dir, labels_file=None, target_size=(512, 512)):
        """
        Initialize dataset handler for CT IQA

        Args:
            data_dir: Directory containing DICOM image files
            labels_file: JSON file with format {"filename": score, ...} e.g., {"0000.dcm": 2.8}
            target_size: Target image size for CNN input
        """

        self.data_dir = data_dir
        self.target_size = target_size
        self.labels_file = labels_file

        if self.labels_file is not None:
            # Load JSON labels
            with open(self.labels_file, "r") as f:
                self.labels_dict = json.load(f)

            # Convert to DataFrame
            self.labels_df = pd.DataFrame(
                [
                    {"filename": filename, "score": score}
                    for filename, score in self.labels_dict.items()
                ]
            )

            self.labels_df["class"] = self.labels_df["score"].apply(self.make_fuzzy_label)

    def make_fuzzy_label(self, value, num_classes=5):
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

    def load_dicom_image(self, filepath):
        """Load and preprocess DICOM image"""

        # Load DICOM file
        dicom_data = pydicom.dcmread(filepath)
        image = dicom_data.pixel_array
        
        # Convert to float32 for processing
        image = image.astype(np.float32)

        # Resize image
        if len(image.shape) == 2:
            image = cv2.resize(image, self.target_size)
            image = np.expand_dims(image, axis=-1)        
        else:
            image = np.array([cv2.resize(slice, self.target_size) for slice in image])
            image = np.expand_dims(image, axis=-1)

        return image
    
    def load_entire_dataset(self):
        images = []
        labels = []

        if self.labels_file is not None:
            for index, row in self.labels_df.iterrows():
                filename = row["filename"]
                label = row["class"]

                filepath = os.path.join(self.data_dir, filename)

                image = self.load_dicom_image(filepath)

                images.append(image)
                labels.append(label)

            return np.array(images), np.array(labels)
        else: 
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                
                image = self.load_dicom_image(filepath)
                images.append(image)

            return np.array(images)

    def test_load_and_display_image(self, images, index, slice_index=None):
        """Test loading and displaying an image"""
        image = images[index]
        
        if len(image.shape) == 3:
            plt.imshow(image[:, :, 0], cmap='gray')
            plt.title(f"Image: {self.labels_df.iloc[index]['filename']}")
        else: 
            if slice_index is None:
                slice_index = image.shape[0] // 2   # Middle slice
            plt.imshow(image[slice_index, :, :, 0], cmap='gray')
            plt.title(f"Slice: {slice_index} from Series: {self.labels_df.iloc[index]['filename']}")

        plt.show()


#%%
