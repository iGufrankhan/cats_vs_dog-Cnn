import tensorflow as tf
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import zipfile
import urllib.request

# 1. Download dataset manually
print("=" * 50)
print("STEP 1: Downloading Cats vs Dogs Dataset...")
print("=" * 50)

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
DATASET_PATH = "kagglecatsanddogs_5340.zip"

if not os.path.exists(DATASET_PATH):
    print("Downloading dataset (this may take a while)...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    print("Download complete!")
else:
    print("Dataset already downloaded.")

# 2. Extract dataset
print("\n" + "=" * 50)
print("STEP 2: Extracting Dataset...")
print("=" * 50)

if not os.path.exists('./PetImages'):
    print("Extracting files...")
    with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Extraction complete!")
else:
    print("Dataset already extracted.")

# 3. Clean corrupted images
print("\n" + "=" * 50)
print("STEP 3: Cleaning Corrupted Images...")
print("=" * 50)

def clean_dataset(base_dir):
    removed_count = 0
    for category in ['Cat', 'Dog']:
        folder = os.path.join(base_dir, category)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                try:
                    # Try to open and verify the image
                    img = tf.keras.preprocessing.image.load_img(filepath)
                    img.close()
                except:
                    # Remove corrupted image
                    os.remove(filepath)
                    removed_count += 1
    return removed_count

removed = clean_dataset('./PetImages')
print(f"Removed {removed} corrupted images")

# 4. Organize dataset
print("\n" + "=" * 50)
print("STEP 4: Organizing Dataset...")
print("=" * 50)

# Move PetImages to cats_vs_dogs/train
if not os.path.exists('./cats_vs_dogs/train'):
    os.makedirs('./cats_vs_dogs/train', exist_ok=True)
    
    # Copy Cat folder
    if os.path.exists('./PetImages/Cat'):
        import shutil
        shutil.copytree('./PetImages/Cat', './cats_vs_dogs/train/cat')
        print("Copied Cat images")
    
    # Copy Dog folder
    if os.path.exists('./PetImages/Dog'):
        import shutil
        shutil.copytree('./PetImages/Dog', './cats_vs_dogs/train/dog')
        print("Copied Dog images")
else:
    print("Dataset already organized")

# Count images
cat_count = len(os.listdir('./cats_vs_dogs/train/cat'))
dog_count = len(os.listdir('./cats_vs_dogs/train/dog'))
print(f"Total images - Cats: {cat_count}, Dogs: {dog_count}")

# 5. Visualize sample images
print("\n" + "=" * 50)
print("STEP 5: Visualizing Sample Images...")
print("=" * 50)

plt.figure(figsize=(10, 5))

cat_images = os.listdir('./cats_vs_dogs/train/cat')[:5]
dog_images = os.listdir('./cats_vs_dogs/train/dog')[:5]

for i, img_name in enumerate(cat_images):
    img_path = os.path.join('./cats_vs_dogs/train/cat', img_name)
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Cat")
    except:
        pass

for i, img_name in enumerate(dog_images):
    img_path = os.path.join('./cats_vs_dogs/train/dog', img_name)
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
        plt.subplot(2, 5, i + 6)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Dog")
    except:
        pass

plt.tight_layout()
plt.savefig('sample_images.png')
print("Sample images saved as 'sample_images.png'")
plt.close()

# 6. Create data generators
print("\n" + "=" * 50)
print("STEP 6: Creating Data Generators...")
print("=" * 50)

datagen = ImageDataGenerator(
    rescale=1/255, 
    validation_split=0.2, 
    rotation_range=10,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1, 
    zoom_range=0.10, 
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    './cats_vs_dogs/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    './cats_vs_dogs/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# 7. Build model
print("\n" + "=" * 50)
print("STEP 7: Building CNN Model...")
print("=" * 50)

model = Sequential()

# 1st layer CNN
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2))

# 2nd layer CNN
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2))

# 3rd Layer
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Flatten())
model.add(Dropout(0.5))

# Fully connected layer
model.add(Dense(512, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 8. Compile model
print("\n" + "=" * 50)
print("STEP 8: Compiling Model...")
print("=" * 50)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9. Train model
print("\n" + "=" * 50)
print("STEP 9: Training Model...")
print("=" * 50)
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# 10. Plot training results
print("\n" + "=" * 50)
print("STEP 10: Plotting Training Results...")
print("=" * 50)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
print("Training results saved as 'training_results.png'")
plt.close()

# 11. Evaluate model
print("\n" + "=" * 50)
print("STEP 11: Evaluating Model...")
print("=" * 50)

val_steps = validation_generator.samples // validation_generator.batch_size + 1
validation_generator.reset()

y_pred = model.predict(validation_generator, steps=val_steps, verbose=1)
y_pred_classes = (y_pred > 0.5).astype(int)
y_true = validation_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Cat', 'Dog']))

# 12. Save model
print("\n" + "=" * 50)
print("STEP 12: Saving Model...")
print("=" * 50)

os.makedirs('model', exist_ok=True)
model.save('model/dog_cat_cnn.h5')
print("Model saved to 'model/dog_cat_cnn.h5'")

# Also save as backup
model.save('cats_vs_dogs.h5')
print("Backup model saved as 'cats_vs_dogs.h5'")

# 13. Test model with online image
print("\n" + "=" * 50)
print("STEP 13: Testing Model with Sample Image...")
print("=" * 50)

try:
    import requests
    from PIL import Image
    
    img_url = "https://hips.hearstapps.com/clv.h-cdn.co/assets/16/18/gettyimages-586890581.jpg"
    img = Image.open(requests.get(img_url, stream=True).raw).resize((150, 150))
    
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(image_array, axis=0)
    img_array = img_array / 255
    
    prediction = model.predict(img_array)
    
    TH = 0.5
    prediction_class = int(prediction[0][0] > TH)
    classes = {v: k for k, v in train_generator.class_indices.items()}
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {classes[prediction_class]} (Confidence: {prediction[0][0]:.2%})")
    plt.savefig('test_prediction.png')
    print(f"Test prediction: {classes[prediction_class]}")
    print(f"Confidence: {prediction[0][0]:.2%}")
    print("Test image saved as 'test_prediction.png'")
    plt.close()
except Exception as e:
    print(f"Could not test with online image: {e}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print(f"Model saved at: model/dog_cat_cnn.h5")
print("You can now run: python app.py")
