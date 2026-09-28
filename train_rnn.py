import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


SEQUENCE_LENGTH = 24
EPOCHS = 10
BATCH_SIZE = 32

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

MODEL_PATH = "models/temperature_rnn.keras"

DATA_URL = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)


np.random.seed(42)
tf.random.set_seed(42)


def load_data():
    print("Loading Jena Climate dataset...")

    zip_path = tf.keras.utils.get_file(
        fname="jena_climate_2009_2016.csv.zip",
        origin=DATA_URL
    )

    data = pd.read_csv(
        zip_path,
        compression="zip"
    )

    data = data[
        [
            "T (degC)",
            "rh (%)",
            "p (mbar)"
        ]
    ]

    print("\nOriginal data:")
    print(data.head())
    print("Original rows:", len(data))

    return data


def convert_to_hourly(data):
    print("\nConverting 10-minute measurements to hourly measurements...")

    hourly_data = data.iloc[::6].reset_index(drop=True)

    print("Hourly rows:", len(hourly_data))

    return hourly_data


def split_data(data):
    total_rows = len(data)

    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(
        total_rows * (
            TRAIN_RATIO + VALIDATION_RATIO
        )
    )

    train_data = data.iloc[:train_end].copy()

    validation_data = data.iloc[
        train_end:validation_end
    ].copy()

    test_data = data.iloc[
        validation_end:
    ].copy()

    print("\nChronological split:")
    print("Train rows:", len(train_data))
    print("Validation rows:", len(validation_data))
    print("Test rows:", len(test_data))

    return train_data, validation_data, test_data


def normalize_data(
    train_data,
    validation_data,
    test_data
):
    train_values = train_data.values.astype(
        np.float32
    )

    validation_values = validation_data.values.astype(
        np.float32
    )

    test_values = test_data.values.astype(
        np.float32
    )

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)

    train_normalized = (
        train_values - mean
    ) / std

    validation_normalized = (
        validation_values - mean
    ) / std

    test_normalized = (
        test_values - mean
    ) / std

    print("\nTraining mean:")
    print(mean)

    print("\nTraining standard deviation:")
    print(std)

    return (
        train_normalized,
        validation_normalized,
        test_normalized,
        mean,
        std
    )


def create_sequences(
    values,
    sequence_length
):
    x = []
    y = []

    for i in range(
        len(values) - sequence_length
    ):
        sequence = values[
            i:i + sequence_length
        ]

        target = values[
            i + sequence_length,
            0
        ]

        x.append(sequence)
        y.append(target)

    return (
        np.array(x, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH, 3)
        ),

        tf.keras.layers.SimpleRNN(
            32,
            activation="tanh"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def plot_training_history(history):
    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        "RNN training and validation loss"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "results/rnn_loss.png"
    )

    plt.close()


def plot_predictions(
    real_values,
    predicted_values
):
    os.makedirs(
        "results",
        exist_ok=True
    )

    samples_to_show = 200

    plt.figure(figsize=(12, 5))

    plt.plot(
        real_values[:samples_to_show],
        label="Real temperature"
    )

    plt.plot(
        predicted_values[:samples_to_show],
        label="Predicted temperature"
    )

    plt.xlabel("Hour")
    plt.ylabel("Temperature (°C)")
    plt.title(
        "Real vs predicted temperature"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "results/rnn_predictions.png"
    )

    plt.close()


def main():
    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    data = load_data()

    data = convert_to_hourly(data)

    (
        train_data,
        validation_data,
        test_data
    ) = split_data(data)

    (
        train_values,
        validation_values,
        test_values,
        mean,
        std
    ) = normalize_data(
        train_data,
        validation_data,
        test_data
    )

    print("\nCreating chronological sequences...")

    x_train, y_train = create_sequences(
        train_values,
        SEQUENCE_LENGTH
    )

    (
        x_validation,
        y_validation
    ) = create_sequences(
        validation_values,
        SEQUENCE_LENGTH
    )

    x_test, y_test = create_sequences(
        test_values,
        SEQUENCE_LENGTH
    )

    print(
        "Train shape:",
        x_train.shape
    )

    print(
        "Validation shape:",
        x_validation.shape
    )

    print(
        "Test shape:",
        x_test.shape
    )

    print(
        "\nEach sequence represents:",
        SEQUENCE_LENGTH,
        "hours"
    )

    print(
        "The model predicts the temperature "
        "of the following hour."
    )

    model = build_model()

    model.summary()

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    )

    print("\nTraining RNN...")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            early_stopping
        ]
    )

    print("\nEvaluating test split...")

    test_loss, test_mae = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(
        f"Test MSE (normalized): "
        f"{test_loss:.4f}"
    )

    print(
        f"Test MAE (normalized): "
        f"{test_mae:.4f}"
    )

    predictions_normalized = (
        model.predict(
            x_test,
            verbose=0
        ).flatten()
    )

    temperature_mean = mean[0]
    temperature_std = std[0]

    real_temperature = (
        y_test * temperature_std
        + temperature_mean
    )

    predicted_temperature = (
        predictions_normalized
        * temperature_std
        + temperature_mean
    )

    mae = mean_absolute_error(
        real_temperature,
        predicted_temperature
    )

    mse = mean_squared_error(
        real_temperature,
        predicted_temperature
    )

    rmse = np.sqrt(mse)

    print(
        f"\nTest MAE: {mae:.2f} °C"
    )

    print(
        f"Test RMSE: {rmse:.2f} °C"
    )

    print("\nExample predictions:")

    for i in range(10):
        print(
            f"Real: "
            f"{real_temperature[i]:.2f} °C "
            f"| Predicted: "
            f"{predicted_temperature[i]:.2f} °C"
        )

    plot_training_history(
        history
    )

    plot_predictions(
        real_temperature,
        predicted_temperature
    )

    model.save(
        MODEL_PATH
    )

    print(
        f"\nModel saved in: "
        f"{MODEL_PATH}"
    )

    print(
        "Loss graph saved in: "
        "results/rnn_loss.png"
    )

    print(
        "Predictions graph saved in: "
        "results/rnn_predictions.png"
    )

    print(
        "\nRNN finished successfully."
    )


if __name__ == "__main__":
    main()