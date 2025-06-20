#%% Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from Input.model_input_fuzzy import CTQualityDataset
from matplotlib import pyplot as plt
from Model.model import AlexNet

# %%
def make_gradcam_heatmap(model, img, last_conv_layer_name, pred_index=None):
    """
    Calculates a GRADCAM map of the last convolutional layer, for one image.

    Args:
        model: trained model
        img: numpy array, shape (1, 512, 512, 1), input image
        last_conv_layer_name: string of the name of last convolutional layer 
                              (can be replaced by any convolutional layer name)
    Returns:
        GRADCAM map as numpy array, shape (512, 512)
    """
    
    input_tensor = tf.keras.Input(shape=(512, 512, 1))
    
    feature_maps = None
    x = input_tensor

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
if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Code/AlexNet_v15_fuzzylabels_b128_LR1e-5_WD1e-3.weights.h5")
        
    val_data_dir = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/Abdomen/images/"
    val_labels_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/Abdomen/val_abdomen_cia.json"
    
    dataset = CTQualityDataset(data_dir=val_data_dir, labels_file=val_labels_file)
    val_images, val_labels = dataset.load_entire_dataset()
    
    #%% 
    index_img = 3
    
    single_img = val_images[index_img]
    single_img_batch = np.expand_dims(single_img, axis=0)
    
    last_conv_layer_name = "conv5"
    
    model.layers[-1].activation = None
    rad_scores = np.argmax(val_labels, axis=1)
    mod_scores = np.argmax(model.predict(val_images, batch_size=128), axis=1)
    heatmap = make_gradcam_heatmap(model, single_img_batch, last_conv_layer_name)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(single_img, cmap="grey")
    ax.matshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")
    plt.title(f"CT + Saliency Overlay (Rad score={rad_scores[index_img]}, Mod score={mod_scores[index_img]})")
    plt.tight_layout()
    plt.show()
# %%
