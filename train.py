import os
import io
import random
import math
import json
import pathlib
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
import logging
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

mixed_precision.set_global_policy("mixed_float16")

physical_gpus = tf.config.experimental.list_physical_devices("GPU")
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU devices: {[gpu.name for gpu in physical_gpus]}")
    except RuntimeError as e:
        print("Error enabling GPU memory growth:", e)
else:
    print("No GPU found or TensorFlow not compiled with GPU support; running on CPU.")

# -------------------- args & constants --------------------
MODEL_DIR = "models"
PARQUET_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE_VAL = 240
LR = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (IMG_SIZE_VAL, IMG_SIZE_VAL)
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- read parquet ------------------------
def load_parquets_to_df(parquet_dir: str) -> pd.DataFrame:
    parquet_files = list(pathlib.Path(parquet_dir).glob("*.parquet"))

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p, engine="pyarrow")
        df["image"] = df["image"].apply(lambda x: x["bytes"])
        df["is_valid"] = df["image"].apply(is_valid_jpeg)
        valid_count = df["is_valid"].sum()
        total_count = len(df)
        print(f"Parquet {p.name}: {valid_count}/{total_count} valid images")
        df = df[df["is_valid"]].drop(columns=["is_valid"])
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise ValueError("No valid images found in Parquet files.")
    df = pd.concat(dfs, ignore_index=True)
    print("DataFrame columns:", df.columns.tolist())
    print("Labels unique values:", df["labels"].unique())
    print(f"Total images after filtering: {len(df)}")
    return df[["image", "labels"]]

def is_valid_jpeg(image_bytes):
    try:
        eoi = image_bytes.rfind(b"\xff\xd9")
        if eoi != -1:
            image_bytes = image_bytes[: eoi + 2]
        tf.io.decode_jpeg(image_bytes, channels=3)
        return True
    except tf.errors.InvalidArgumentError:
        return False

df = load_parquets_to_df(PARQUET_DIR)
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["labels"], random_state=42)

# -------------------- tf.data pipeline --------------------

def decode_img(x):
    img = tf.io.decode_jpeg(x, channels=3, try_recover_truncated=True, acceptable_fraction=0.7)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def make_ds(dataframe, shuffle: bool = True) -> tf.data.Dataset:
    img_bytes = dataframe["image"].values
    labels = dataframe["labels"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((img_bytes, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_bytes), reshuffle_each_iteration=True)

    ds = ds.map(lambda x, y: (decode_img(x), y), num_parallel_calls=AUTOTUNE, deterministic=False)

    ds = ds.cache()

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(train_df)
val_ds = make_ds(val_df, shuffle=False)

# -------------------- model -------------------------------
base = tf.keras.applications.EfficientNetB1(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = tf.keras.layers.RandomContrast(0.1)(x)
x = tf.keras.layers.RandomBrightness(0.1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
model = tf.keras.Model(inputs, outputs)

class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["labels"])
cw = {0: class_weights[0], 1: class_weights[1]}

steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)

lr_schedule = CosineDecayRestarts(
    initial_learning_rate=LR,
    first_decay_steps=steps_per_epoch,
    t_mul=2.0,
    m_mul=0.8,
    alpha=1e-5,
)

optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_auc"),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "catdog_best.keras"), save_best_only=True, monitor="val_auc"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=cw,
    callbacks=callbacks,
)

base.trainable = True

for layer in base.layers[:-20]:
    layer.trainable = False

ft_lr_schedule = CosineDecayRestarts(
    initial_learning_rate=LR / 10,
    first_decay_steps=steps_per_epoch,
    t_mul=2.0,
    m_mul=0.8,
    alpha=1e-5,
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=ft_lr_schedule, weight_decay=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
ft_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
)