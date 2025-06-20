# %% Imports
import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
import numpy as np
from Input.model_input_fuzzy import CTQualityDataset
from matplotlib import pyplot as plt
import pickle
from Data_Augmentation.data_augmentation import DataAugmentatie

# %% Class definition

class AlexNet(Sequential):
    """ 
    Model Architecture for Image Quality Assessment of CT images. Based on AlexNet Architecture.

    This class implements a convolutional neural network inspired by the AlexNet architecture, 
    which is tailored for the task of image quality assessment.

    Attributes: 
        None
    
    Example: 
        To use this model, instantiate it by using the following script:
        
        ''' python
        model = AlexNet()
        history = model.fit(train_images, train_labels, epochs=..., batch size=..., ...)
        '''
    
    Note: 
        Ensure that the images are of shape (512, 512, 1) as grayscale.
    """
    def __init__(self):
        """
        Initialize and compile the AlexNet model.
        """

        super().__init__()
        self.add(keras.layers.InputLayer(shape=(512, 512, 1), name="input_layer_custom"))
        
        self.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", data_format="channels_last", activation="relu", kernel_initializer="he_normal", name="conv1"))
        self.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", data_format="channels_last", name="pool1"))
        
        self.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_initializer="he_normal", name="conv2"))
        self.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", data_format="channels_last", name="pool2"))
        
        self.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_initializer="he_normal", name="conv3"))
        
        self.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_initializer="he_normal", name="conv4"))
        
        self.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last", activation="relu", kernel_initializer="he_normal", name="conv5"))
        self.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", data_format="channels_last", name="pool3"))
        
        self.add(keras.layers.GlobalAveragePooling2D(name="gap"))

        self.add(keras.layers.Dense(units=1024, activation="relu", kernel_initializer="he_normal", name="dense1"))
        self.add(keras.layers.Dropout(rate=0.5, name="dropout1"))
        
        self.add(keras.layers.Dense(units=512, activation="relu", kernel_initializer="he_normal", name="dense2"))
        self.add(keras.layers.Dropout(rate=0.3, name="dropout2"))

        self.add(keras.layers.Dense(units=5, activation="softmax", kernel_initializer="glorot_uniform", name="dense3"))
        
        self.compile(                                               
            optimizer=tf.keras.optimizers.AdamW(1e-5, weight_decay=1e-3),          
            loss="categorical_crossentropy",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse"),
                     keras.metrics.CategoricalAccuracy(name="cat_accuracy")
                    ]
        )

# %% Model fitting
if __name__ == "__main__": 
    train_data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/image"
    test_data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC_test/LDCTIQAC_test/images"
    train_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json"
    test_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC_test/LDCTIQAC_test/test.json"

    train_dataset = CTQualityDataset(train_data_dir, train_file)
    test_dataset = CTQualityDataset(test_data_dir, test_file)
    train_images, train_labels = train_dataset.load_entire_dataset()
    test_images, test_labels = test_dataset.load_entire_dataset()
    
    rotation_range = 15
    translation_range = 10
    num_augmentations = 1

    augmenteren = DataAugmentatie(rotation_range, translation_range)

    train_images_aug, train_labels_aug = augmenteren.augment_dataset(train_images, train_labels, num_augmentations)
    model = AlexNet()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="AlexNet_v15_fuzzylabels_b128_LR1e-5_WD1e-3.weights.h5", 
            save_best_only=True, save_weights_only=True, monitor="val_cat_accuracy", mode="max"
            ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
            )
    ]

#%%
    history = model.fit(
        train_images_aug, 
        train_labels_aug,
        batch_size=128,
        epochs=80,
        validation_data=(test_images, test_labels),
        callbacks=callbacks
    )

    train_history_dict = history.history
    with open("train_history_AlexNet_v15_fuzzylabels_b128_LR1e-5_WD1e-3_dict", "wb") as file:
        pickle.dump(train_history_dict, file)

# %%
    print(f"Epoch: {np.argmin(history.history['val_rmse'])+1}, val_rmse: {min(history.history['val_rmse'])}")
    print(f"Epoch: {np.argmin(history.history['rmse'])+1}, rmse: {min(history.history['rmse'])}")
    print(f"Epoch: {np.argmin(history.history['val_loss'])+1}, Val_loss: {min(history.history['val_loss'])}")
    print(f"Epoch: {np.argmin(history.history['loss'])+1}, loss: {min(history.history['loss'])}")
    print(f"Epoch: {np.argmax(history.history['val_cat_accuracy'])+1}, Val_cat_accuracy: {max(history.history['val_cat_accuracy'])}")
    print(f"Epoch: {np.argmax(history.history['cat_accuracy'])+1}, cat_accuracy: {max(history.history['cat_accuracy'])}")

# %% Plotting Epoch-Accuracy curve & Epoch-Loss curve
    
    cat_accuracy = history.history["cat_accuracy"]
    val_cat_accuracy = history.history["val_cat_accuracy"]
    epochs = range(1, len(cat_accuracy) + 1)

    plt.plot(epochs, cat_accuracy, marker="o")
    plt.plot(epochs, val_cat_accuracy, marker="o")
    plt.title("cat_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("cat_accuracy")
    plt.legend(["train", "test"])
    plt.show()

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.plot(epochs, loss, marker="o")
    plt.plot(epochs, val_loss, marker="o")
    plt.title("model_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"])
    plt.show()
# %%
