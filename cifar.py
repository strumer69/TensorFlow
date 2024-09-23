import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from datetime import datetime


# Check if TensorFlow can detect GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from using all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0  # Normalize the data
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Create a simple CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

def scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.1
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Log start time for training
start_time = time.time()

# Fit the model
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          validation_data=(x_test, y_test),
          epochs=2,
          callbacks=[checkpoint, lr_scheduler])

# Log training time
training_time = time.time() - start_time

# Log start time for evaluation
eval_start_time = time.time()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

# Log evaluation time
evaluation_time = time.time() - eval_start_time
current_time = datetime.now().strftime("%d %b %Y %H:%M:%S")

# Print the results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Log the runtime, results, and date into a text file
with open("runtime_log.txt", "a") as f:  # Use "a" to append to the file
    f.write(f"Run Date and Time: {current_time}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write("\n")  # Optional: Add a new line between records



