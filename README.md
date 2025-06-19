# Delta-IQA: A Convolutional Neural Network for Automatic Quality Assessment of CT Images
Introduction... 

## Model Architecture 
![Diagram](Images/Framework.png)

## Requirements 
- Python 3.10.16
- Tensorflow 2.18.1
- Keras 3.6.0
- Numpy 2.0.2
- Scipy 1.15.3
- Pandas 2.2.3
- Matplotlib 3.10.3
- OpenCV 4.11.0.86
- Scikit-learn 1.6.1
- Scikit-image 0.25.2

## Dataset
The LDCTIQAC 2023 dataset used for training, testing and internal validation is available [here](https://ldctiqac2023.grand-challenge.org).  
Images from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/), using scores from the Leiden University Medical Centre, were used for external validation. 

## Performance of the model
On the internal validation set, the overall accuracy is 0.67. 


## Credits
Created by: Lars Jongsma, Koen Walet, Joey Mulder & Thomas Dijkstra (2025).
