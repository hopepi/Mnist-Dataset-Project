import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.datasets import mnist
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

data_train = {"Images": X_train, "Labels": Y_train}
data_test = {"Images": X_test, "Labels": Y_test}

label_train = np.argmax(data_train["Labels"], axis=1)
label_test = np.argmax(data_test["Labels"], axis=1)

plt.subplot(1, 2, 1)
label_train_counts = pd.Series(label_train).value_counts()
sns.barplot(x=label_train_counts.index, y=label_train_counts.values, palette='Blues',hue=label_train_counts.index, legend=False)
plt.xlabel("Number")
plt.ylabel("Label Count")
plt.xticks(label_train_counts.index)
plt.title("Train number and label count figure")

plt.subplot(1, 2, 2)
label_test_counts = pd.Series(label_test).value_counts()
sns.barplot(x=label_test_counts.index, y=label_test_counts.values, palette='Reds',hue=label_train_counts.index, legend=False)
plt.xlabel("Number")
plt.ylabel("Label Count")
plt.xticks(label_test_counts.index)
plt.title("Test number and label count figure")
plt.show()


plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[label_train == i][0], cmap='gray')  # İlk örneği göster
    plt.title(f'Class {i}')
    plt.axis('off')
plt.show()


_, img_height, img_width, channel = X_train.shape
batch_size = 32


train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.1, horizontal_flip=True, rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size)


model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation="relu", padding="same", input_shape=(img_height, img_width, channel)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (2, 2), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (2, 2), activation="relu"),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


steps_per_epoch = train_generator.n // batch_size
validation_steps = test_generator.n // batch_size


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=test_generator,
    validation_steps=validation_steps
)


test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test doğruluğu: {test_acc}")



plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)  # 1. grafik
plt.plot(history.history["loss"], label="Loss of Education")
plt.title("Loss Of Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)  # 2. grafik
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.title("Accuracy Of Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation")
plt.legend()

plt.subplot(2, 2, 3)  # 3. grafik
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Val Loss Of Epoch")
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 4)  # 4. grafik
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Val Accuracy Of Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation")
plt.legend()

plt.tight_layout()
plt.show()
