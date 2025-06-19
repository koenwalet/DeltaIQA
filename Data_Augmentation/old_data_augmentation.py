#%%
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os 
from skimage.transform import rotate, warp, EuclideanTransform, AffineTransform

#%% Load data


class data_inladen_dataframe:
    def __init__(self, data_path, labels_file, target_size=(512,512,)):
        # initialiseer alle objecten die je gaat gebruiken in deze class.

        self.data_path = data_path
        self.target_size = target_size

        # load JSON labels
        with open(labels_file, 'r') as f:
            self.labels_path = json.load(f)

        # Convert to DataFrame
        self.labels_dataframe = pd.DataFrame([{'filename': filename, 'score': score}
                                              for filename, score in self.labels_path.items()])
        
        self.labels_dataframe['class'] = self.labels_dataframe['score'].apply(self.make_fuzzy_label)

    def make_fuzzy_label(self, value, num_classes=5):
        """Converts continuous labels to fuzzy labels"""
        
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
        # inladen en preprocessing images

        # Load TIF file
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # Convert to float 32 for processing
        image = image.astype(np.float32)

        # Indien nodig resize image
        image = cv2.resize(image, self.target_size)

        # Indien nodig, zorg dat image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add a channel dimension to make ik 512x512x1
        image = np.expand_dims(image, axis=-1)

        return image
    
    def load_entire_dataset(self):  # Door self slaat hij alles op uit deze functie. 
        images = []
        labels = []

        for index, row in self.labels_dataframe.iterrows():
            filename = row['filename']
            label = row['class']

            # Construct full filepath
            filepath = os.path.join(self.data_path, filename)

            # Load the image
            image = self.load_tif_image(filepath)

            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)
    
    def test_load_and_display_image(self, images, index):
        # Display the image
        plt.imshow(images[index, :, :, :], cmap='gray')
        plt.title(f'Image: {self.labels_dataframe.iloc[index]["filename"]}')
        plt.show()




#%%
# Data augmenteren met translaties en rotaties

class DataAugmentatie: 
    def __init__(self, rotation_range, translation_range):
        # Data augmentatie met skimage
    
        self.rotation_range = rotation_range
        self.translation_range = translation_range


    # Functies voor het roteren, transleren, combi en flippen.
    def rotate_image(self, image, angle):
        # Roteer de image met een gegeven hoek
        rotated = rotate(image[:,:,0], angle, resize=False, preserve_range=True, mode='constant', cval=0)

        # Zorg dat float32 blijft en breng de 512x512x1 weer terug bij return
        rotated = rotated.astype(np.float32)
        return np.expand_dims(rotated, axis=-1)

    def translate_image(self, image, vector, order=1):
        # Maak transformatie matrix
        transformatie = EuclideanTransform(translation=vector)

        # Pas transformatie toe
        translated = warp(image[:,:,0], transformatie, order=order, preserve_range=True, mode='constant', cval=0)
        # Order: interpolatie order (0=nearest, 1=linear, 2=quadratic, 3=cubic)

        # Zorg dat float32 blijft en breng de 512x512x1 weer terug bij return
        translated = translated.astype(np.float32)
        return np.expand_dims(translated, axis=-1)

    def combined_rotate_and_translate_image(self, image, angle, vector, order=1):
        # Maak de transformatie matrix via affine
        transformatie_affine = AffineTransform(
            rotation=np.radians(angle),
            translation=vector
        )

        # Pas de transformatie toe
        combi_affine = warp(image[:,:,0], transformatie_affine, order=order, preserve_range=True, mode='constant', cval=0)

        # Zorg dat float32 blijft en breng de 512x512x1 weer terug bij return
        combi_affine = combi_affine.astype(np.float32)
        return np.expand_dims(combi_affine, axis=-1)
    
    def flip_x_as(self, image):
        # flippen image horizontaal over de x-as, dit via 180 graden draaien beeld.
        flipped = rotate(image[:, :, 0], 180, resize=False, preserve_range=True, mode='constant', cval=0)
        
        # Zorg dat float32 blijft en breng de 512x512x1 weer terug bij return
        flipped = flipped.astype(np.float32)
        return np.expand_dims(flipped, axis=-1)

    
    # Augmenteren van single image
    def augment_single_image(self, image, number_augmentations, order=1):
        augmented_images = []

        for _ in range(number_augmentations):  # de _ geeft aan dat dit variabele _ niet gebruikt wordt in de for-loop voor de rest.
            # Augmenteren met rotaties random
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            aug_image_rot = self.rotate_image(image, angle)
            augmented_images.append(aug_image_rot)
        
            # Augmenteren met translaties via vectors random
            shift_x = np.random.randint(-self.translation_range, self.translation_range)
            shift_y = np.random.randint(-self.translation_range, self.translation_range)
            vector = (shift_x, shift_y)

            aug_image_trans = self.translate_image(image, vector, order)
            augmented_images.append(aug_image_trans)

            # Augmenteren met combi van rotaties en translaties
            aug_image_combi = self.combined_rotate_and_translate_image(image, angle, vector, order)
            augmented_images.append(aug_image_combi)

            # Deleten variabelen om geheugen vrij te maken voor afbeeldingen verwerken of opslaan.
            del aug_image_rot, aug_image_trans, aug_image_combi
        
        # Augmenteren met flippen over x-as, deze buiten for-loop omdat maar 1x hoeft.
        # aug_image_flip = self.flip_x_as(image)
        # augmented_images.append(aug_image_flip)

        # del aug_image_flip

        return augmented_images

    # Augmenteren van de hele dataset
    def augment_dataset(self, images, labels, number_augmentations, batch_size=10):
        # lijst aanmaken voor geaugmenteerde images met originele images ook. Hetzelfde voor de labels
        augmented_images = list(images)
        augmented_labels = list(labels)

        # Per batch size doen, zodat geheugen bespaart
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
        
            # Voor elke originele image augmenteren
            for image, label in zip(batch_images, batch_labels):
                # Voer de augmentaties uit
                aug_images = self.augment_single_image(image, number_augmentations)

                # Voeg geaugmenteerde images toe aan dataset met bijbehorende label
                for aug_image in aug_images:
                    augmented_images.append(aug_image)
                    augmented_labels.append(label)

        print(f"Originele dataset grootte: {len(images)} images")
        print(f"Nieuwe geaugmenteerde dataset grootte: {len(augmented_images)} images and {len(augmented_labels)} labels")
        print(f"Dataset vergroot met factor: {len(augmented_images) / len(images):.1f}")
        #print(augmented_labels)

        return np.array(augmented_images), np.array(augmented_labels)       

    # Test 1 image voor controleren augmentations
    def visualize_augmentations(self, image, label):
        fig, axes = plt.subplots(2, 8, figsize=(20,8))

        # Originele image
        axes[0,0].imshow(image[:,:,0], cmap='gray')
        axes[0,0].set_title(f'Origineel\nLabel: {label}')
        axes[0,0].axis('off')

        # Augmented image
        aug_images = self.augment_single_image(image, number_augmentations=2)

        for i, aug_image in enumerate(aug_images):
            axes[1,i].imshow(aug_image[:,:,0], cmap='gray')
            axes[1,i].set_title(f'Augmentatie {i+1}\nLabel: {label}')
            axes[1,i].axis('off')

        plt.tight_layout()
        plt.show()



#%%
if __name__ == "__main__":
    # Deel data inladen
    data_path = "C:/Users/Joey/Downloads/LDCTIQAG2023_train/LDCTIQAG2023_train/image"
    labels_file = "C:/Users/Joey/Downloads/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json"

    dataset = data_inladen_dataframe(data_path, labels_file)

    images, labels = dataset.load_entire_dataset()

    #print(images, labels)
    #print(images.shape)

    #dataset.test_load_and_display_image(images=images, index=0)

    # Deel augmenteren vaste waarden
    # rotatie_angles = [-20, -15, -10, 10, 15, 20]
    # translatie_vectors = [
    #     (20, 0),   # 10 pixels naar rechts
    #     (-20, 0),  # 10 pixels naar links  
    #     (0, 20),   # 10 pixels naar beneden
    #     (0, -20),  # 10 pixels naar boven
    #     (20, 20),    # 5 pixels diagonaal rechts-beneden
    #     (-20, -20),  # 5 pixels diagonaal links-boven
    # ]

    #augmenteren = DataAugmentatie(rotatie_angles, translatie_vectors)

    # Deel augmenteren random waarden
    rotatie_range = 15
    translatie_range = 10
    aantal_augmentaties = 1

    augmenteren = DataAugmentatie(rotatie_range, translatie_range)

    augmenteren.augment_dataset(images, labels, aantal_augmentaties)
    print('Augmenteren volledige dataset gelukt :)')

    # print('Test augmentatie:')
    # augmenteren.visualize_augmentations(images[0], labels[0])



#           # Progressie indicator
                # if i % 100 == 0: 
                #     print(f"Augmenteren image {i}/{len(images)}")
