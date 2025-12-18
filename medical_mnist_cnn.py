import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "medical_mnist"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
RANDOM_SEED = 42
EPOCHS = 15

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = RESULTS_DIR / "medical_mnist_cnn.keras"

# =====================================================================
# 2. DATA LOADING
# =====================================================================

def load_datasets():
    """Load images from DATA_DIR and create training and validation datasets."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {DATA_DIR}\n"
            "Verify that the Medical MNIST dataset is extracted here."
        )

    print(f"Using images from: {DATA_DIR}")

    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    print(f"Number of classes: {len(class_names)}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names

# =====================================================================
# 3. MODEL
# =====================================================================

def build_model(num_classes):
    """Build a CNN for 64x64x3 images."""
    inputs = keras.Input(shape=IMG_SIZE + (3,))

    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.1)(x)

    x = keras.layers.Rescaling(1.0 / 255)(x)

    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="medical_mnist_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# =====================================================================
# 4. PLOTTING
# =====================================================================

def plot_training_history(history, out_dir: pathlib.Path):
    """Save training and validation accuracy and loss curves."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    # Accuracy curve
    plt.figure()
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(out_dir / "accuracy.png")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(out_dir / "loss.png")
    plt.close()

    print(f"Saved training curves to {out_dir}")

def plot_confusion_matrices(cm, class_names, out_dir: pathlib.Path):
    """Save confusion matrix and normalized confusion matrix as images."""
    # Raw confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Normalized)")

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_normalized.png")
    plt.close(fig)

    print(f"Saved confusion matrices to {out_dir}")

# =====================================================================
# 5. EVALUATION
# =====================================================================

def evaluate_model(model, val_ds, class_names, out_dir: pathlib.Path):
    """Evaluate on the validation set and save text and image results."""
    print("\nEvaluating on validation data...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    all_labels = []
    all_preds = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds_labels)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\nClassification Report:")
    print(report)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nValidation accuracy: {val_acc:.4f}, validation_loss: {val_loss:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrices(cm, class_names, out_dir)

# =====================================================================
# 6. MAIN
# =====================================================================

def main():
    print("Loading Medical MNIST dataset...")
    train_ds, val_ds, class_names = load_datasets()

    print("\nBuilding CNN model...")
    model = build_model(num_classes=len(class_names))

    print("\nTraining model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    print("\nSaving model...")
    model.save(MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

    print("\nPlotting training history...")
    plot_training_history(history, RESULTS_DIR)

    evaluate_model(model, val_ds, class_names, RESULTS_DIR)

    print("\nFinished. Results directory:", RESULTS_DIR)


if __name__ == "__main__":
    main()
