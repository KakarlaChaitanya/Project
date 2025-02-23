from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your image data paths (modify as needed)
train_data_dir = "D:\svcet"
validation_data_dir = "D:\svcet"
#test_data_dir = "path/to/your/test/images"  # Optional for evaluation

# Image dimensions (modify if your images have different sizes)
img_width, img_height = 150, 150

# Data augmentation (optional, can improve model robustness)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training and validation data with augmentation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'  # Adjust for your classification task
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN model (modify based on your needs)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Adjust num_classes for your task

# Compile the model (adjust loss function, optimizer, and metrics as needed)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model (optional)
# test_datagen = ... (create data generator for test data if needed)
# score = model.evaluate(test_datagen, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# Save the model (optional)
model.save('your_model.h5')