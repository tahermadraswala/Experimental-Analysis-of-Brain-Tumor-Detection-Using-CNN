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

Data preparation

Key image processing parameters used in the code:

IMG_SIZE = 224 (images resized to 224×224) — compatible with VGG16

BATCH_SIZE = 32

train_datagen uses augmentation: rotation, shifts, shear, zoom, horizontal flip, and validation_split=0.2.

test_datagen only rescales with 1./255.

The script creates three generators using flow_from_directory:

train_generator (subset='training')

validation_generator (subset='validation')

test_generator (shuffle=False)

Model architecture

The model is built with the following design:

VGG16 base (ImageNet weights, include_top=False, input_shape=(224,224,3))

Freeze base model for stage 1 training

Add top layers:

GlobalAveragePooling2D()

BatchNormalization()

Dropout(0.5)

Dense(512, activation='relu')

BatchNormalization()

Dropout(0.3)

Dense(num_classes, activation='softmax')

This configuration balances pre-trained feature extraction with a moderately-sized classification head.

Training procedure

Training is split into two stages.

Stage 1 — Transfer Learning (Frozen base):

Freeze VGG16 base (base_model.trainable = False)

Optimizer: Adam(learning_rate=0.001)

Loss: categorical_crossentropy

Metrics: accuracy

Epochs used in script: INITIAL_EPOCHS = 5 (adjustable)

Stage 2 — Fine-Tuning:

Unfreeze VGG16 and freeze layers up until block5_conv1 (i.e. only deeper blocks and top classifier trainable)

Optimizer: Adam(learning_rate=1e-5) (lower LR for fine-tuning)

Additional epochs: FINE_TUNE_EPOCHS = 5 (total epochs = INITIAL + FINE_TUNE)

Why two stages?

Stage 1 lets the new classification head learn without disrupting pre-trained weights.

Stage 2 lets the deeper convolutional layers refine features to the MRI domain while retaining earlier learned features.
