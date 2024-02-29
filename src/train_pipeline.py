import sys
import os
sys.path.append('components')

from components.data_loader import AutoEncoderDataset, ClassifierDataset
from components.models import AutoEncoder, Classifier
from components.model_trainer import ModelTrainer
from utils import visualise_reconstruction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import warnings

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_autoencoder():
    # Data loading
    train = AutoEncoderDataset(type_data="train")
    val = AutoEncoderDataset(type_data="val")
    test = AutoEncoderDataset(type_data="test")

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True)

    # Hyperparameters
    autoencoder = AutoEncoder().to(training_parameters['device'])
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    training_parameters = {
        "model": autoencoder,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": nn.MSELoss(),
        "optimizer": autoencoder_optimizer,
        "lr_scheduler": optim.lr_scheduler.StepLR(autoencoder_optimizer, step_size=5, gamma=0.5),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Training
    EPOCHS = 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autoencoder_trainer = ModelTrainer(**training_parameters)
        autoencoder_trainer.fit(EPOCHS)

    # Visualisations and feedback on training
    visualisation_dir = 'visualisation/autoencoder'
    ensure_dir(visualisation_dir)
    autoencoder_trainer.plot_training_history(save_path=f'{visualisation_dir}/training_history.png')
    test_loader = DataLoader(test, batch_size=100, shuffle=True)
    visualise_reconstruction(autoencoder, test_loader, save_path=f'{visualisation_dir}/reconstruction.png')

    return autoencoder

def train_classifier(autoencoder):
    # Data Loading
    train = ClassifierDataset(type_data="train")
    val = ClassifierDataset(type_data="val")
    test = ClassifierDataset(type_data="test")

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=True)

    # Hyperparameters
    classifier = Classifier(autoencoder, 10).to(training_parameters['device'])
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    training_parameters = {
        "model": classifier,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": classifier_optimizer,
        "lr_scheduler": optim.lr_scheduler.StepLR(autoencoder_optimizer, step_size=5, gamma=0.3),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Training
    EPOCHS = 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier_trainer = ModelTrainer(**training_parameters)
        classifier_trainer.fit(EPOCHS)

    # Visualisations and feedback on training
    visualisation_dir = 'visualisation/classifier'
    ensure_dir(visualisation_dir)
    classifier_trainer.plot_training_history(save_path=f'{visualisation_dir}/training_history.png')

    # Test Performance
    test_loader = DataLoader(test, batch_size=100, shuffle=True)
    classifier_trainer.testing_and_plotting(test_loader, save_path=f'{visualisation_dir}/test_performance.png')

if __name__ == "__main__":
    autoencoder = train_autoencoder()
    train_classifier(autoencoder)