import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path


def load_model():
    model_path = "catdog_best.keras" if Path("models/catdog_best.keras").exists() else "catdog_best.h5"
    if not Path(model_path).exists():
        print("❌ No model found (catdog_best.keras or catdog_best.h5).")
        exit(1)
    print(f"✅ Loading {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def predict(image_path):
    model = load_model()
    _, height, width, _ = model.input_shape

    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(height, width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    print(f"🔮 Predicting {image_path}")
    prob = model.predict(img_array, verbose=0)[0, 0]
    label = "dog" if prob >= 0.5 else "cat"
    print(f"Result: {label} (probability: {prob:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", default="dog.jpg", help="Path to image")
    predict(parser.parse_args().image_path)