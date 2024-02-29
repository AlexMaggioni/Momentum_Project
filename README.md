# FashionMNIST Semi-Supervised Learning with Convolutional Autoencoder

This project presents a semi-supervised learning pipeline for classifying images from the FashionMNIST dataset into 10 distinct clothing categories. Utilizing a minimal amount of labeled data, the solution employs a convolutional autoencoder for feature extraction, followed by a classification stage that operates on these extracted features.

## Methodology Overview

The approach is split into two distinct phases. Initially, a convolutional autoencoder is employed to learn meaningful features from the dataset in an unsupervised manner through the task of image reconstruction. Subsequently, these learned features are used to train a classifier, leveraging a small subset (10%) of labeled data, showcasing the efficiency of semi-supervised learning techniques.

### Convolutional Autoencoder for Feature Learning

The autoencoder architecture is designed with residual blocks, enabling it to effectively encode crucial aspects of the input images. This stage aims to minimize the reconstruction error, thereby facilitating the learning of a compact and informative latent representation.

### Classifier Training with Limited Label Usage

The encoder part of the autoencoder transforms images into latent representations, which are then used to train a classifier. This process exemplifies how semi-supervised learning can mitigate the need for extensive labeled datasets.

## Implementation Details

### Object-Oriented Principles and Modularity

The codebase is structured following object-oriented programming (OOP) principles to enhance readability, maintainability, and scalability. Key components of the pipeline, such as models (`models.py`), data loaders (`data_loader.py`), and the model training logic (`model_trainer.py`), are modularized. This organization allows for a clear separation of concerns, making the codebase easier to navigate and modify.

### Streamlined Pipeline

The entire training pipeline is streamlined within `train_pipeline.py`, which orchestrates the training process from data loading and preprocessing to model training and evaluation. This centralized script ensures a cohesive workflow, leveraging the modular components.

### Interactive Notebooks

For a more intuitive understanding of the project and its codebase, a Jupyter notebook is included. This notebook provide step-by-step guidance through the project's key phases, offering insights into the methodology, code execution, and results visualization.

## Getting Started

### Prerequisites

- torch
- torchvision
- matplotlib
- seaborn
- numpy
- scikit-learn

### Installation

Clone this repository and install the dependencies:

```bash
git clone <repository-url>
cd <repository-path>
pip install -r requirements.txt
```

### Visualizations

The project generates visualizations to illustrate learning progress and model performance, including loss curves and accuracy trends. These are saved in the `visualisation/` directory.

## File Descriptions

- `README.md`: Overview and guide for the project.
- `requirements.txt`: Lists dependencies.
- `src/`: Source code, including:
  - Jupyter Notebooks for more intuitive approach.
  - `train_pipeline.py`: Centralized training and evaluation script.
  - `utils.py`: Visualization utilities.
  - `components/`: Core ML components (data loaders, model definitions, training logic).

## Reflections and Future Directions

Acknowledging the advancements in semi-supervised learning, particularly through state-of-the-art contrastive autoencoders like SimCLR, I recognize the potential for achieving superior performance with such approaches. However, due to time constraints and the limitations of my computing resources (I have a modest laptop, haha), I opted for simpler techniques in this project.

If the use of labels was not permitted, I would have leveraged methodologies akin to those described in the SCAN paper ([Learning to Classify Images without Labels](https://arxiv.org/abs/2005.12320)). This approach suggests performing clustering on the latent representations derived from the autoencoder, using these clusters as proxy labels.
