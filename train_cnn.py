import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("Loading Beans dataset...")
dataset = load_dataset("AI-Lab-Makerere/beans")

class_names = dataset["train"].features["labels"].names
print("Classes:", class_names)


def prepare_split(split):
    images = []
    labels = []

    for example in split:
        image = example["image"].convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.asarray(image, dtype=np.float32) / 255.0

        images.append(image)
        labels.append(example["labels"])

    return np.asarray(images), np.asarray(labels)


print("Preparing train split...")
x_train, y_train = prepare_split(dataset["train"])

print("Preparing validation split...")
x_validation, y_validation = prepare_split(dataset["validation"])

print("Preparing test split...")
x_test, y_test = prepare_split(dataset["test"])

print("Train shape:", x_train.shape)
print("Validation shape:", x_validation.shape)
print("Test shape:", x_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("Evaluating test split...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)
display.plot(xticks_rotation=20)
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cnn_accuracy.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cnn_loss.png"), dpi=150)
plt.close()

model_path = os.path.join(MODEL_DIR, "beans_cnn.keras")
model.save(model_path)

print(f"Model saved in: {model_path}")
print("CNN finished successfully.")
