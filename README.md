# Butterfly Species Classification 🦋

## Introduction
This project focuses on the classification of butterfly species using deep learning techniques. I utilize two powerful convolutional neural network architectures, ResNet50 and MobileNet, 
to accurately identify butterfly species from a dataset of labeled images.

## Dataset
* The dataset used in this project consists of 5000 labeled images of 75 different butterfly species. 
* The label of each image are saved in Training_set.csv.
* The Testing_set.csv contains names of image in test folder.

## Hyperparameters
* Batch size: 32
* Epochs: 23
* Learning rate: 0.0001

## Data Preprocessing
1. Load and preprocess images: Resize to 240x240.
2. Normalized by dividing pixel values by 255.0
3. Labels are extracted from a DataFrame (train_df).
4. Labels are converted to categorical format.
5. Split dataset: 70% training, 15% validation, 15% testing.


## Model Training
1. ResNet50:
* Load pre-trained ResNet50.
* Freeze first 20% of layers.
* Optimizer: Adam.
* Dense layer with 256 units and ReLU activation.
* Output layer with 75 units and softmax activation.


2. MobileNetV2 Model
* MobileNetV2 is loaded with pre-trained weights.
* All layers are frozen initially.
* Dense layer with 512 units and ReLU activation.
* Output layer with 75 units and softmax activation.
* Optimizer: Adam

## Evaluation and Results
1. ResNet50 Model
* Training time: 7.65 hours
* Test Accuracy: 78.80%

2. MobileNetV2 Model
* Training Time: 1.65 hours
* Test Accuracy: 84.40%


## Conclusion
Overall, MobileNetV2 outperformed ResNet50 in both training efficiency and test accuracy for this image classification task, making it a strong candidate for practical applications.

## Future Work
* Exploring other architectures like EfficientNet or DenseNet.
* Fine-tuning the hyperparameters for better performance.

## Contact
For any questions or suggestions, please contact Viet at your phamvietofficial@gmail.com
