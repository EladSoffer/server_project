import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint

# Step 1: Load Pre-trained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Step 2: Freeze the Convolutional Base
base_model.trainable = False
class_names = os.listdir('train')
num_classes = len(class_names)

# Step 3: Add Custom Top Layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # num_classes the number of the fruits
])

# Step 4: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Data Augmentation and Preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Vary brightness to simulate different lighting conditions
    channel_shift_range=0.1  # Slightly change the colors
)

train_generator = datagen.flow_from_directory(
    'train',  # Replace with the actual path to your data
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Step 6: ModelCheckpoint to Save the Best Model
checkpoint = ModelCheckpoint('best_fruit_model_vgg15.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Step 7: Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=25,
    validation_data=train_generator,  # You may use a separate validation generator if you have validation data
    validation_steps=len(train_generator) // 5,
    callbacks=[checkpoint]
)

# Optional: Step 8 - Evaluate the Model
# This is how you would evaluate the model after training, assuming you have a test set
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc:.2f}')
