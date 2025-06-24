import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import logging

# -------------------- Setup --------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)

# -------------------- Constants --------------------
MODEL_DIR = "models"
PARQUET_DIR = "data"
IMG_SIZE_VAL = 240
BATCH_SIZE = 16
IMG_SIZE = (IMG_SIZE_VAL, IMG_SIZE_VAL)
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- Data Loading --------------------
def is_valid_jpeg(image_bytes):
    try:
        eoi = image_bytes.rfind(b'\xff\xd9')
        if eoi != -1:
            image_bytes = image_bytes[: eoi + 2]
        tf.io.decode_jpeg(image_bytes, channels=3)
        return True
    except tf.errors.InvalidArgumentError:
        return False

def load_parquets_to_df(parquet_dir: str) -> pd.DataFrame:
    parquet_files = list(pathlib.Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in directory {parquet_dir}")

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p, engine="pyarrow")
        df["image"] = df["image"].apply(lambda x: x["bytes"])
        df["is_valid"] = df["image"].apply(is_valid_jpeg)
        df = df[df["is_valid"]].drop(columns=["is_valid"])
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        raise ValueError("No valid images found in Parquet files.")
        
    df = pd.concat(dfs, ignore_index=True)
    return df[["image", "labels"]]

# -------------------- TF.data Pipeline --------------------
def decode_img(x):
    img = tf.io.decode_jpeg(x, channels=3, try_recover_truncated=True, acceptable_fraction=0.7)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def make_ds(dataframe) -> tf.data.Dataset:
    img_bytes = dataframe["image"].values
    labels = dataframe["labels"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((img_bytes, labels))
    ds = ds.map(lambda x, y: (decode_img(x), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# -------------------- Evaluation Functions --------------------
def plot_confusion_matrix(y_true, y_pred_classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix saved to confusion_matrix.png")

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    print("✅ ROC curve saved to roc_curve.png")


def main():
    # --- Load Model ---
    model_path = pathlib.Path(MODEL_DIR) / "catdog_best.keras"
    if not model_path.exists():
        print(f"? No model found at {model_path}")
        return
    print(f"? Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # --- Load Data ---
    print("?? Loading and processing validation data...")
    df = load_parquets_to_df(PARQUET_DIR)
    _, val_df = train_test_split(df, test_size=0.15, stratify=df["labels"], random_state=42)
    val_ds = make_ds(val_df)
    y_true = val_df["labels"].values

    # --- Evaluation ---
    print("?? Evaluating model...")
    loss, accuracy, auc_metric = model.evaluate(val_ds, verbose=1)
    print(f"\n? Evaluation Results:")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - AUC: {auc_metric:.4f}")

    # --- Predictions for Detailed Metrics ---
    print("\n?? Generating predictions for detailed metrics...")
    y_pred_probs = model.predict(val_ds, verbose=1).flatten()
    y_pred_classes = (y_pred_probs >= 0.5).astype(int)

    # --- Classification Report ---
    print("\n? Classification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=["Cat", "Dog"])
    print(report)

    # --- Plotting ---
    print("?? Generating plots...")
    plot_confusion_matrix(y_true, y_pred_classes)
    plot_roc_curve(y_true, y_pred_probs)

if __name__ == "__main__":
    main()
