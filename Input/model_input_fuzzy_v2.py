# %%
# Imports
import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
class CTQualityDataset:
    def __init__(self, data_dir, labels_file=None, target_size=(512, 512)):
        """
        Initialize dataset handler for CT IQA

        Args:
            data_dir: Directory containing TIF image files
            labels_file: JSON file with format {"filename": score, ...} e.g., {"0000.tif": 2.8}
            target_size: Target image size for CNN input
        """

        self.data_dir = data_dir
        self.target_size = target_size
        self.labels_file = labels_file

        if self.labels_file is not None:
            # Load JSON labels
            with open(labels_file, "r") as f:
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

    def load_tif_image(self, filepath):
        """Load and preprocess TIF image"""

        # Load TIF file
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        # Convert to float32 for processing
        image = image.astype(np.float32)

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add a channel dimension to make it 512x512x1
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

                image = self.load_tif_image(filepath)

                images.append(image)
                labels.append(label)

            return np.array(images), np.array(labels)
        else: 
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                
                image = self.load_tif_image(filepath)
                images.append(image)

            return np.array(images)

    def test_load_and_display_image(self, images, index):
        """Test loading and displaying an image"""

        # Display the image
        plt.imshow(images[index, :, :, :], cmap='gray')
        plt.title(f"Image {index}")
        plt.show()

# %%
if __name__ == "__main__": 
    data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/image"
    labels_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json"
    
    dataset = CTQualityDataset(data_dir, labels_file)

    images, labels = dataset.load_entire_dataset()

    print(images, labels)
    print(images.shape)

    dataset.test_load_and_display_image(images=images, index=0)

    print(dataset.labels_df)
# %%
