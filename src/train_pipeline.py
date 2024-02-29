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

    print("-----------------------------------")
    print("-- Autoencoder Training Pipeline --")
    print("-----------------------------------")

    # Data loading
    print("Step 1: Data Loading")
    train = AutoEncoderDataset(type_data="train")
    val = AutoEncoderDataset(type_data="val")
    test = AutoEncoderDataset(type_data="test")

    train_loader = DataLoader(train, batch_size=100, shuffle=True)
    val_loader = DataLoader(val, batch_size=100, shuffle=True)
    test_loader = DataLoader(test, batch_size=100, shuffle=True)

    print("Step 2: Hyperparameters Set")
    # Hyperparameters
    autoencoder = AutoEncoder()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    training_parameters = {
        "model": autoencoder,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": nn.MSELoss(),
        "optimizer": autoencoder_optimizer,
        "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(autoencoder_optimizer, mode='min', factor=0.1, patience=3, min_lr=0.000001),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    autoencoder.to(training_parameters['device'])

    print("Step 3: Training Model")
    # Training
    EPOCHS = 50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autoencoder_trainer = ModelTrainer(**training_parameters)
        autoencoder_trainer.fit(EPOCHS)
    
    print("Step 4: Visualisations and feedback on training")
    # Visualisations and feedback on training
    visualisation_dir = 'visualisation/autoencoder'
    ensure_dir(visualisation_dir)
    autoencoder_trainer.plot_training_history(save_path=f'{visualisation_dir}/training_history.png')
    test_loader = DataLoader(test, batch_size=100, shuffle=True)
    visualise_reconstruction(autoencoder, test_loader, save_path=f'{visualisation_dir}/reconstruction.png')

    return autoencoder

def train_classifier(autoencoder):

    print("-----------------------------------")
    print("-- Classifier Training Pipeline --")
    print("-----------------------------------")

    print("Step 1: Data Loading")
    # Data Loading
    train = ClassifierDataset(type_data="train")
    val = ClassifierDataset(type_data="val")
    test = ClassifierDataset(type_data="test")

    train_loader = DataLoader(train, batch_size=100, shuffle=True)
    val_loader = DataLoader(val, batch_size=100, shuffle=True)
    test_loader = DataLoader(test, batch_size=100, shuffle=True)
    
    print("Step 2: Hyperparameters Set")
    # Hyperparameters
    classifier = Classifier(autoencoder, 10)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    training_parameters = {
        "model": classifier,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": classifier_optimizer,
        "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min', factor=0.1, patience=3, min_lr=0.000001),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    classifier.to(training_parameters['device'])


    print("Step 3: Training Model")
    # Training
    EPOCHS = 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier_trainer = ModelTrainer(**training_parameters)
        classifier_trainer.fit(EPOCHS)
    
    print("Step 4: Visualisations and feedback on training\n")
    # Visualisations and feedback on training
    visualisation_dir = 'visualisation/classifier'
    ensure_dir(visualisation_dir)
    classifier_trainer.plot_training_history(save_path=f'{visualisation_dir}/training_history.png')

    # Test Performance
    test_loader = DataLoader(test, batch_size=100, shuffle=True)
    classifier_trainer.testing_and_plotting(test_loader, save_path=f'{visualisation_dir}/test_performance.png')

if __name__ == "__main__":
    test = AutoEncoder()
    autoencoder = train_autoencoder()
    train_classifier(test)