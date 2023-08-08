import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

folder_path = os.path.join(os.path.expanduser('~'), 'wildlifedata')
# define path to data
base_dir = folder_path

# create an image data generator object
datagen = ImageDataGenerator(rescale=1./255)

# load training data
train_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'train'),
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32
)

print(train_generator.class_indices)

# load the validation data
validation_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'valid'),
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32
)

""" # load test data
test_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'valid'),
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=16
) """

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-3:]:
    layer.trainable = True

model = Sequential([
    base_model,
    Dropout(.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(.25),
    Dense(1, activation='sigmoid')
])

""" model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
]) """

optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=3
)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a new ImageDataGenerator for the misclassification analysis
datagen = ImageDataGenerator(rescale=1./255)

# Create a new validation generator with shuffle=False
validation_generator = datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'valid'),
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=False  # This is important!
)

# Get the true labels from the generator
y_true = validation_generator.classes

# Make predictions for all validation images
y_pred = model.predict(validation_generator)
y_pred = np.where(y_pred > 0.5, 1, 0).flatten()  # adjust the threshold as needed

# Compute the validation accuracy
accuracy = np.mean(y_true == y_pred)
print(f'Validation accuracy: {accuracy * 100:.2f}%')

# Find the indices of the misclassified images
misclassified_indices = np.where(y_true != y_pred)[0]

# Get the directory of the current .py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path to the 'errors' directory
errors_dir = os.path.join(current_dir, 'errors')

# Initialize the current index
current_index = 0

# Iterate over the validation generator
for images, labels in validation_generator:
    # Iterate over each image in the batch
    for i in range(len(images)):
        # If the current index is in the misclassified indices, save the image
        if current_index in misclassified_indices:
            # Get the corresponding image and label
            image = images[i]
            true_label = labels[i]
            pred_label = y_pred[current_index]
            
            # Save the image to a file
            plt.imshow(image)
            plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
            plt.savefig(os.path.join(errors_dir, f'misclassified_image_{current_index}.png'))
            plt.clf()  # Clear the current figure so the next plot doesn't overlap with this one
        
        # Increment the current index
        current_index += 1