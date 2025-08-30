# Install dependencies (quiet mode -q to reduce logs)
!pip install -q kaggle tensorflow scikit-learn matplotlib seaborn opencv-python

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


print("=== AI-Powered Brain Tumor Detection System ===")
print("Classifying: Glioma, Meningioma, Pituitary, No Tumor")
print("=" * 50)

# Upload kaggle.json only if not already uploaded
if not os.path.exists('/content/kaggle.json'):
    print("Upload kaggle.json (download it from your Kaggle account settings > API):")
    files.upload()

# Setup Kaggle API
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
!chmod 600 /content/kaggle.json

# Download dataset
if not os.path.exists('/content/brain-tumor-mri-dataset.zip'):
    print("Downloading Brain Tumor MRI Dataset...")
    !kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
else:
    print("Dataset already downloaded.")

# Extract dataset
if not os.path.exists('/content/Training'):
    with zipfile.ZipFile('/content/brain-tumor-mri-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("✅ Dataset extracted successfully!")
else:
    print("✅ Dataset already extracted.")

# Define dataset paths
TRAIN_PATH = '/content/Training'
TEST_PATH = '/content/Testing'

# Class labels
CLASSES = sorted(os.listdir(TRAIN_PATH))
num_classes = len(CLASSES)
print(f"Classes to classify: {CLASSES}")

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Training + validation generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

def create_brain_tumor_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    return model

# Build and compile
model = create_brain_tumor_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_brain_tumor_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]
# Stage 1 - Transfer Learning
print("\n🔥 Stage 1: Training with Frozen Base Model")
INITIAL_EPOCHS = 5

history_stage1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Stage 2 - Fine-Tuning
print("\n🎯 Stage 2: Fine-Tuning VGG16 Layers")
base_model = model.layers[0]
base_model.trainable = True

fine_tune_from = 'block5_conv1'
for layer in base_model.layers:
    if layer.name == fine_tune_from:
        break
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

history_stage2 = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history_stage1.epoch[-1] + 1,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

def combine_histories(hist1, hist2):
    combined_history = {}
    for key in hist1.history.keys():
        combined_history[key] = hist1.history[key] + hist2.history[key]
    return combined_history

full_history = combine_histories(history_stage1, history_stage2)

def plot_training_history(history, stage1_epochs):
    acc, val_acc = history['accuracy'], history['val_accuracy']
    loss, val_loss = history['loss'], history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    plt.axvline(x=stage1_epochs, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.legend(); plt.title('Accuracy'); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.axvline(x=stage1_epochs, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.legend(); plt.title('Loss'); plt.grid(True)
    plt.show()

plot_training_history(full_history, INITIAL_EPOCHS)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n🎯 Test Accuracy: {test_accuracy*100:.2f}% | Test Loss: {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

model.save("brain_tumor_detection_finetuned_model.h5")
print("✅ Final model saved as brain_tumor_detection_finetuned_model.h5")

