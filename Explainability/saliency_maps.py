#%% Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import numpy as np
from model_input_fuzzy import CTQualityDataset
from matplotlib import pyplot as plt
from data_augmentation import DataAugmentatie
from AlexNet_model_v6_fuzzylabels import AlexNet
from IPython.display import Image, display

# %%
model = AlexNet()
model.load_weights("best_AlexNet_model_ordinal_continuous_v1.weights.h5")


train_data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/image"
train_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json"
train_dataset = CTQualityDataset(train_data_dir, train_file)
train_images, train_labels = train_dataset.load_entire_dataset()

# %%
single_img = train_images[10]
single_img_batch = np.expand_dims(single_img, axis=0)

#%%
def compute_saliency_map(model, img, class_index):
    """
    Calculates a saliency map for one image and class index.

    Args:
        model: trained model
        img: numpy array, shape (1, 512, 512, 1), input image
        class_index: int, desired class index for saliency map

    Returns:
        saliency map as numpy array, shape (512, 512)
    """

    img = tf.convert_to_tensor(img)
    img = tf.cast(img, tf.float32)
    img = tf.Variable(img)

    with tf.GradientTape() as tape:
        tape.watch(img)
        preds = model(img)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, img)
    grads = tf.abs(grads)
    
    grads -= tf.reduce_min(grads)
    grads /= (tf.reduce_max(grads) + 1e-8)

    saliency = grads[0, :, :, 0]
    return saliency.numpy()

# %%
saliency_map = compute_saliency_map(model, single_img_batch, 0)
plt.imshow(saliency_map, cmap='hot')
plt.axis('off')
plt.show()


# %%
def make_gradcam_heatmap(model, img, last_conv_layer_name, pred_index=None):
    input_tensor = tf.keras.Input(shape=(512, 512, 1))

    x = input_tensor
    feature_maps = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            feature_maps = x

    output = x

    grad_model = keras.models.Model(inputs=input_tensor, outputs=[feature_maps, output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_ouput = last_conv_layer_output[0]
    heatmap = last_conv_layer_ouput @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap[..., tf.newaxis]
    return tf.image.resize(heatmap.numpy(), (512, 512))


# %%
last_conv_layer_name = "conv5"

model.layers[-1].activation = None
preds = model.predict(single_img_batch)
heatmap = make_gradcam_heatmap(model, single_img_batch, last_conv_layer_name)

fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(single_img, cmap="grey")
ax.matshow(heatmap, cmap="jet", alpha=0.4)
ax.axis("off")
plt.title("CT + Saliency Overlay")
plt.tight_layout()
plt.show()
# %%
