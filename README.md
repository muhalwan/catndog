# 🐱🐶 Cat vs. Dog Classifier (EfficientNet-B0, Keras/TensorFlow)

A lightweight CNN that predicts whether an image contains **a cat or a dog**.
The backbone is `EfficientNetB0` pre-trained on ImageNet and fine-tuned on the
[microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
training split (23 410 images). You can get the model at https://huggingface.co/deruppu/catndog

## Model Details

|                         | Value |
|-------------------------|-------|
| Backbone                | EfficientNet-B0 (`include_top=False`) |
| Input size              | `128×128×3` |
| Extra layers            | GlobalAvgPool ➜ Dropout(0.2) ➜ Dense(1, **sigmoid**) |
| Precision               | Mixed-precision (`float16` activations / `float32` dense) |
| Optimizer               | **AdamW** with cosine-decay-restarts schedule |
| Loss                    | Binary Cross-Entropy |
| Epochs                  | 25 (frozen backbone) + 5 (fine-tune full network) |
| Batch size              | 16 |
| Class weighting         | Balanced weights computed from training labels |

### Validation Metrics

| Metric      | Value |
|-------------|-------|
| Accuracy    | **97.2 %** |
| AUC         | **0.9967** |
| Loss (BCE)  | 0.079 |

*(computed on 15 % stratified validation split – 3 512 images)*

## Intended Uses & Limitations

* **Intended** : quick demos, tutorials, educational purposes, CAPTCHA-like tasks.
* **Not intended** : production-grade pet breed classification, safety-critical
  applications.
* The model only distinguishes **cats** vs **dogs**; images with neither are
  undefined behaviour.
* Trained on 128×128 crops; very large images might require resizing first.

## Dataset Credits

The training data is the publicly available
[microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
dataset (originally the Asirra CAPTCHA dataset). **Huge thanks** to Microsoft
Research and Petfinder.com for releasing the images!

```
@misc{microsoftcatsdogs,
  title  = {Cats vs. Dogs Image Dataset},
  author = {Microsoft Research & Petfinder.com},
  howpublished = {HuggingFace Hub},
  url    = {https://huggingface.co/datasets/microsoft/cats_vs_dogs}
}
```

## Acknowledgements

* TensorFlow/Keras team for the excellent deep-learning framework.
* Mingxing Tan & Quoc V. Le for EfficientNet.
* The Hugging Face community for the awesome Model & Dataset hubs.
