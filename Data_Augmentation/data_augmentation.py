#%%
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, warp, EuclideanTransform, AffineTransform
from Input.model_input_fuzzy import CTQualityDataset

#%% 
class DataAugmentation: 
    """Dataset augmentation with translations and rotations"""
    def __init__(self, rotation_range, translation_range):
        """Initialize dataset augmentation.
        Args: 
            rotation_range: range for calculated rotation angles
            translation_range" range for possible variation in calculated coordinates"""
        self.rotation_range = rotation_range
        self.translation_range = translation_range


    def rotate_image(self, image, angle):
        """Rotate image with given angle"""
        rotated = rotate(image[:,:,0], angle, resize=False, preserve_range=True, mode='constant', cval=0)

        rotated = rotated.astype(np.float32)
        return np.expand_dims(rotated, axis=-1)


    def translate_image(self, image, vector, order=1):
        """Translate image with given vector of translation parameters for x- and y-axis"""
        transformation_euclidean = EuclideanTransform(translation=vector)
        translated = warp(image[:,:,0], transformation_euclidean, order=order, preserve_range=True, mode='constant', cval=0)

        translated = translated.astype(np.float32)
        return np.expand_dims(translated, axis=-1)


    def combined_rotate_and_translate_image(self, image, angle, vector, order=1):
        """Rotation and translation transformation of an image for given rotation angle and vector of translation parameters for x- and y-axis"""
        transformation_affine = AffineTransform(
            rotation=np.radians(angle),
            translation=vector
        )

        combi_affine = warp(image[:,:,0], transformation_affine, order=order, preserve_range=True, mode='constant', cval=0)

        combi_affine = combi_affine.astype(np.float32)
        return np.expand_dims(combi_affine, axis=-1)
    

    def augment_single_image(self, image, number_augmentations, order=1):
        """Augmentation of a single image for a number of augmentations. Returns list of augmented images"""
        augmented_images = []

        for _ in range(number_augmentations):  
            # Random rotation angle between given range
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            aug_image_rot = self.rotate_image(image, angle)
            augmented_images.append(aug_image_rot)
        
            # Random translation in x- and y-axis between given range
            shift_x = np.random.randint(-self.translation_range, self.translation_range)
            shift_y = np.random.randint(-self.translation_range, self.translation_range)
            vector = (shift_x, shift_y)
            aug_image_trans = self.translate_image(image, vector, order)
            augmented_images.append(aug_image_trans)

            # Both rotation and translation
            aug_image_combined = self.combined_rotate_and_translate_image(image, angle, vector, order)
            augmented_images.append(aug_image_combined)

            del aug_image_rot, aug_image_trans, aug_image_combined
        return augmented_images


    def augment_dataset(self, images, labels, number_augmentations, batch_size=10):
        """Augmentation of whole dataset (multiple images). Returns array of original image and augmented images.
        Args:
            images: input array of images
            labels: input array of labels
            number_agumentations: number of augmentations per wanted per image
            batch_size=10: step size, runs augmentation process in batches 
        """
        augmented_images = list(images)
        augmented_labels = list(labels)

        # Runs augmentation per batch for saving memory on computer
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
        
            # Augment for single image
            for image, label in zip(batch_images, batch_labels):
                aug_images = self.augment_single_image(image, number_augmentations)

                for aug_image in aug_images:
                    augmented_images.append(aug_image)
                    augmented_labels.append(label)

        print(f"Original dataset size: {len(images)} images")
        print(f"New augmented dataset size: {len(augmented_images)} images and {len(augmented_labels)} labels")
        print(f"Dataset enlarged with factor: {len(augmented_images) / len(images):.1f}")

        return np.array(augmented_images), np.array(augmented_labels)       

    
    def visualize_augmentations(self, image, label):
        """Test 2 augmentations on single image"""
        fig, axes = plt.subplots(2, 8, figsize=(20,8))

        # Original image
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

    data_path = "C:/Users/Joey/Downloads/LDCTIQAG2023_train/LDCTIQAG2023_train/image"
    labels_file = "C:/Users/Joey/Downloads/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json"

    dataset = CTQualityDataset(data_path, labels_file)

    images, labels = dataset.load_entire_dataset()

    # Augmentation with range of values
    rotation_range = 15
    translation_range = 10
    number_of_augmentations = 1

    augmentation = DataAugmentation(rotation_range, translation_range)

    augmentation.augment_dataset(images, labels, number_of_augmentations)
    print('Augmentation of dataset complete')

