# Experimental-Analysis-of-Brain-Tumor-Detection-Using-CNN
Overview

This project implements an experimental pipeline to classify brain MRI images into four classes: Glioma, Meningioma, Pituitary, and No Tumor. It uses transfer learning with a pre-trained VGG16 backbone (ImageNet weights), a two-stage training process (frozen base + fine-tuning), data augmentation, and standard evaluation metrics (accuracy, classification report, confusion matrix).

The included code automates dataset download (Kaggle), extraction, generator creation, model build/compile, training, fine-tuning, evaluation, and model saving.

Features

Automatic dataset download using Kaggle API (if kaggle.json is provided)

ImageDataGenerator pipelines for training/validation/test with augmentation

Transfer learning with VGG16 (frozen base then fine-tuned starting from block5_conv1)

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Training history plotting (accuracy & loss)

Classification report & confusion matrix visualization

Saves final fine-tuned model to disk
