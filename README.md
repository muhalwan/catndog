# Cat vs. Dog Image Classifier

This repository contains the code for a Cat vs. Dog image classifier built with TensorFlow and Keras. The model uses the `EfficientNetB1` architecture and is trained to distinguish between images of cats and dogs.

## Features

- **Model**: `EfficientNetB1` pre-trained on ImageNet.
- **Framework**: TensorFlow / Keras.
- **Training**: Includes transfer learning and fine-tuning stages.
- **Data Augmentation**: Uses a variety of data augmentation techniques to improve model robustness.
- **Scripts**: Provides scripts for training (`train.py`), evaluation (`eval.py`), and inference (`test.py`).

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- Pandas
- Scikit-learn

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/muhalwan/catndog.git
    cd catndog
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data

The model is trained on a Parquet dataset of cat and dog images. Place your `.parquet` files in the `data/` directory.

## Training

To train the model, run the `train.py` script:

```bash
python train.py
```

The script will perform the following steps:
1.  Load the Parquet files from the `data/` directory.
2.  Split the data into training and validation sets.
3.  Build the `EfficientNetB1` model.
4.  Train the model using transfer learning and fine-tuning.
5.  Save the best model to the `models/` directory.

## Evaluation

To evaluate the trained model, run the `eval.py` script:

```bash
python eval.py
```

This will load the best model and compute performance metrics on the test set.

## Results

The model achieves the following performance on the test set:

| Metric   | Value  |
|----------|--------|
| Loss     | 0.1495 |
| Accuracy | 98.75% |
| AUC      | 0.9986 |

## Acknowledgements

- The training data is from the [microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs) dataset.
- This project is built upon the excellent work of the TensorFlow/Keras team and the creators of EfficientNet.
