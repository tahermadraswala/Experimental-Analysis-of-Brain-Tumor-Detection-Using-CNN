# Brain Tumor Detection Using CNN 

# Install required packages
!pip install -q kaggle tensorflow scikit-learn matplotlib seaborn

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files

# ==============================================================================
# STEP 1: Kaggle Dataset Setup 
# ==============================================================================

# Upload kaggle.json file
print("Upload your kaggle.json file:")
files.upload()

# Move kaggle.json to the correct location
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Create a datasets folder
!mkdir -p datasets

# Download the Kaggle dataset
!kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p datasets

# Unzip the dataset
!unzip -q datasets/*.zip -d datasets

# Verify the dataset directory
!ls datasets
!pwd
!ls

# ==============================================================================
# STEP 2: Load Testing Data 
# ==============================================================================

# Define the path to the TESTING dataset first 
dataset_path = "/content/datasets/Testing"
categories = sorted(os.listdir(dataset_path))  # Sort for consistency

print(f"Categories found: {categories}")

# Load testing data
data = []
labels = []

# Iterate through each category folder
for category in categories:
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue
        
    label = categories.index(category)  # Assign labels (0, 1, etc.)
    print(f"Processing {category} with label {label}")

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        try:
            # Load and preprocess the image
            img = load_img(img_path, target_size=(128, 128))  # Resize as needed
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

# Convert data and labels to NumPy arrays
data = np.array(data) / 255.0  # Normalize pixel values
labels = np.array(labels)

print("Testing dataset loaded successfully!")
print(f"Total images: {len(data)}, Total labels: {len(labels)}")

# ==============================================================================
# STEP 3: Load Training Data 
# ==============================================================================

# Define the path to the TRAINING dataset
train_dataset_path = "/content/datasets/Training"
train_categories = sorted(os.listdir(train_dataset_path))

print(f"Training categories found: {train_categories}")

# Load training data
train_data = []
train_labels = []

# Iterate through each category folder
for category in train_categories:
    category_path = os.path.join(train_dataset_path, category)
    if not os.path.isdir(category_path):
        continue
        
    label = train_categories.index(category)  # Assign labels
    print(f"Processing training {category} with label {label}")

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        try:
            # Load and preprocess the image
            img = load_img(img_path, target_size=(128, 128))  # Resize as needed
            img_array = img_to_array(img)
            train_data.append(img_array)
            train_labels.append(label)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

# Convert data and labels to NumPy arrays
train_data = np.array(train_data) / 255.0  # Normalize pixel values
train_labels = np.array(train_labels)

print("Training dataset loaded successfully!")
print(f"Total training images: {len(train_data)}, Total training labels: {len(train_labels)}")

# ==============================================================================
# STEP 4: Combined Dataset Processing
# ==============================================================================

# Process both training and testing sets together for consistency
train_path = "/content/datasets/Training"
test_path = "/content/datasets/Testing"

# Process the training set
categories_train = sorted(os.listdir(train_path))
final_train_data = []
final_train_labels = []

for category in categories_train:
    category_path = os.path.join(train_path, category)
    if not os.path.isdir(category_path):
        continue
        
    label = categories_train.index(category)  # Assign a label
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = load_img(img_path, target_size=(128, 128))  # Resize images
                img_array = img_to_array(img)
                final_train_data.append(img_array)
                final_train_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

# Process the testing set
categories_test = sorted(os.listdir(test_path))
final_test_data = []
final_test_labels = []

for category in categories_test:
    category_path = os.path.join(test_path, category)
    if not os.path.isdir(category_path):
        continue
        
    label = categories_test.index(category)  # Assign a label
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = load_img(img_path, target_size=(128, 128))  # Resize images
                img_array = img_to_array(img)
                final_test_data.append(img_array)
                final_test_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

# Convert to NumPy arrays
final_train_data = np.array(final_train_data) / 255.0  # Normalize pixel values
final_train_labels = np.array(final_train_labels)
final_test_data = np.array(final_test_data) / 255.0    # Normalize pixel values
final_test_labels = np.array(final_test_labels)

print("Training and testing datasets prepared successfully!")
print(f"Training data: {len(final_train_data)}, Testing data: {len(final_test_data)}")

# ==============================================================================
# STEP 5: Data Visualization 
# ==============================================================================

# Display the first 5 images from the training set
print("Training set samples:")
for i in range(min(5, len(final_train_data))):
    plt.figure(figsize=(6, 4))
    plt.imshow(final_train_data[i])
    plt.title(f"Training Label: {final_train_labels[i]} ({categories_train[final_train_labels[i]]})")
    plt.axis('off')  # Remove axes for better display
    plt.show()

# Display the first 5 images from the testing set
print("Testing set samples:")
for i in range(min(5, len(final_test_data))):
    plt.figure(figsize=(6, 4))
    plt.imshow(final_test_data[i])
    plt.title(f"Testing Label: {final_test_labels[i]} ({categories_test[final_test_labels[i]]})")
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 6: Data Splitting (Properly Using Separate Train/Test Sets)
# ==============================================================================

# Use the separate training and testing datasets as provided by Kaggle
# Training set: Split into train (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    final_train_data, final_train_labels, test_size=0.2, random_state=42, stratify=final_train_labels
)

# Testing set: Use the separate test data as final test set
X_test = final_test_data
y_test = final_test_labels

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")
print(f"Total samples: {len(X_train) + len(X_val) + len(X_test)}")

print("\nDataset Distribution:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")

# ==============================================================================
# STEP 7: Label Encoding 
# ==============================================================================

# One-hot encoding
num_classes = len(categories_train)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(f"Label shape after encoding: {y_train.shape}")
print(f"Number of classes: {num_classes}")

# ==============================================================================
# STEP 8: CNN Model Creation (Following Original Architecture)
# ==============================================================================

# Create the CNN model (following original structure)
model = Sequential([
    # Convolutional layers
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten and fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Dynamic number of classes
])

model.summary()  # Print model architecture

# ==============================================================================
# STEP 9: Model Compilation 
# ==============================================================================

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==============================================================================
# STEP 10: Model Training 
# ==============================================================================

print("Starting model training...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,  s
                    batch_size=32,
                    verbose=1)

# ==============================================================================
# STEP 11: Model Evaluation 
# ==============================================================================

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==============================================================================
# STEP 12: Training History Visualization 
# ==============================================================================

# Visualize training accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 13: Save Model 
# ==============================================================================

model.save("brain_tumor_detection_model.h5")
print("Model saved as brain_tumor_detection_model.h5")

# ==============================================================================
# STEP 14: Data Augmentation Setup 
# ==============================================================================

# Data augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ==============================================================================
# STEP 15: Predictions and Evaluation 
# ==============================================================================

# Predict on test images
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Calculate accuracy on the test set
test_accuracy_manual = np.sum(predicted_classes == actual_classes) / len(actual_classes)
print(f"Manual Test Accuracy: {test_accuracy_manual * 100:.2f}%")

# Display a few predictions
print("Sample predictions:")
for i in range(min(5, len(X_test))):
    plt.figure(figsize=(6, 4))
    plt.imshow(X_test[i])
    plt.title(f"Actual: {actual_classes[i]} ({categories_train[actual_classes[i]]})\nPredicted: {predicted_classes[i]} ({categories_train[predicted_classes[i]]})")
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 16: Training History Validation 
# ==============================================================================

# Plot training and validation accuracy 
if 'history' in locals():
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Training history not found. Make sure to assign the history object when training the model.")

# ==============================================================================
# STEP 17: Detailed Confusion Matrix 
# ==============================================================================

# Validate array shapes
print(f"Predicted labels shape: {predicted_classes.shape}")
print(f"Actual labels shape: {actual_classes.shape}")

# Ensure predicted_labels and actual_labels have the same length
if len(predicted_classes) != len(actual_classes):
    print(f"Length mismatch: Predicted ({len(predicted_classes)}), Actual ({len(actual_classes)})")
else:
    # Compute confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=categories_train, yticklabels=categories_train)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.show()

# ==============================================================================
# STEP 18: Classification Report 
# ==============================================================================

# Print classification report
print("Classification Report:")
print(classification_report(actual_classes, predicted_classes, target_names=categories_train))

# Show additional test predictions using the separate test set
print("Additional test predictions on separate test set:")
for i in range(min(5, len(final_test_data))):
    plt.figure(figsize=(6, 4))
    plt.imshow(final_test_data[i])
    
    # Make prediction on this image
    test_img = np.expand_dims(final_test_data[i], axis=0)
    test_pred = model.predict(test_img)
    test_pred_class = np.argmax(test_pred)
    
    plt.title(f"Actual: {categories_test[final_test_labels[i]]}\nPredicted: {categories_train[test_pred_class]}")
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 19: Enhanced Data Augmentation 
# ==============================================================================

# Enhanced data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator on training data
datagen.fit(final_train_data)

# ==============================================================================
# STEP 20: Transfer Learning Setup 
# ==============================================================================

# Load a pre-trained model without the top layer 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers
base_model.trainable = False

# Create transfer learning model
transfer_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')  # Use dynamic number of classes
])

# Compile the transfer learning model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Transfer learning model created successfully!")
print(f"Transfer model can be trained using: transfer_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)")

# ==============================================================================
# STEP 21: Single Image Prediction Function 
# ==============================================================================

def predict_new_image(model, img_path, categories):
    """
    Function to predict a single new image (following original logic)
    """
    try:
        # Load and preprocess the new image
        img = load_img(img_path, target_size=(128, 128))  # Resize to model input size
        img_array = img_to_array(img) / 255.0  # Normalize pixel values

        # Predict the class
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class = np.argmax(prediction, axis=1)
        
        print(f"Predicted class: {categories[predicted_class[0]]}")
        
        # Plot the image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {categories[predicted_class[0]]}")
        plt.axis('off')
        plt.show()
        
        return predicted_class[0]
    
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

# Example usage (uncomment when you have a test image):
# predicted_class = predict_new_image(model, "/content/your_test_image.jpg", categories_train)

print("\n" + "="*80)
print("BRAIN TUMOR DETECTION MODEL COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Categories: {categories_train}")
print(f"Model saved as: brain_tumor_detection_model.h5")
print("="*80)
