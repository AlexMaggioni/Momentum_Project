import torch
import os
import pickle
import sys
from models import AutoEncoder, Classifier

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device):
        """
        Initializes the ModelTrainer class with the model, data loaders, criterion, optimizer, lr_scheduler, and device.
        """
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Create training_artifacts folder if it doesn't exist
        os.makedirs("training_artifacts", exist_ok=True)

        # Initialize lists to track loss history
        self.train_loss_history = []
        self.val_loss_history = []

    def train_one_epoch(self):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        self.train_loss_history.append(epoch_loss)
        print(f'Training Loss: {epoch_loss:.4f}')

    def validate(self):
        """
        Evaluates the model on the validation dataset.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)
        self.val_loss_history.append(epoch_loss)
        print(f'Validation Loss: {epoch_loss:.4f}')
        return epoch_loss

    def fit(self, epochs):
        
        if self.model.__class__.__name__ == "AutoEncoder":
            model_filename = "AutoEncoder_best.pth"
        elif self.model.__class__.__name__ == "Classifier":
            model_filename = "Classifier_best.pth"
        else:
            raise ValueError("Model type not recognized. Please provide a valid model type.")

        model_path = os.path.join("training_artifacts", model_filename)

        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}\n{"-"*30}')
            self.train_one_epoch()
            val_loss = self.validate()
            self.lr_scheduler.step()

            # Save the model if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, model_path)  
                print(f"Model updated with improved validation loss: {best_val_loss:.4f}")