import torch
import os
from models import AutoEncoder, Classifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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

        # Initialize lists to track loss and accuracy history
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []  

    def train_one_epoch(self):
        """
        Trains the model for one epoch, computing the training accuracy if the model is a Classifier.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

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
            
            if isinstance(self.model, Classifier):
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        self.train_loss_history.append(epoch_loss)
        
        if isinstance(self.model, Classifier):
            epoch_accuracy = 100 * correct_predictions / total_predictions
            self.train_accuracy_history.append(epoch_accuracy)
            print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')
        else:
            print(f'Training Loss: {epoch_loss:.4f}')

    def validate(self):
        """
        Evaluates the model on the validation dataset, computing the validation accuracy if the model is a Classifier.
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                
                if isinstance(self.model, Classifier):
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += targets.size(0)
                    correct_predictions += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        self.val_loss_history.append(epoch_loss)
        
        if isinstance(self.model, Classifier):
            epoch_accuracy = 100 * correct_predictions / total_predictions
            self.val_accuracy_history.append(epoch_accuracy)
            print(f'Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.2f}%')
        else:
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
            self.lr_scheduler.step(val_loss)

            # Save the model if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved with improved validation loss: {best_val_loss:.4f}")

    def plot_training_history(self):
        """
        Plots the training and validation loss, as well as accuracy (if applicable),
        over the epochs to visualize the progress of training.
        """
        epochs = range(1, len(self.train_loss_history) + 1)
        
        # Plot training and validation loss
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, 'r-', label='Training Loss')
        plt.plot(epochs, self.val_loss_history, 'b-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training and validation accuracy if we have accuracy data
        if self.train_accuracy_history and self.val_accuracy_history:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.train_accuracy_history, 'r-', label='Training Accuracy')
            plt.plot(epochs, self.val_accuracy_history, 'b-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def testing_and_plotting(self, test_loader):
        """
        Evaluates the model on the test dataset, plots performance metrics
        including accuracy, precision, recall, F1 score, and plots an aesthetic confusion matrix.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Plotting metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x=metrics, y=values, palette="viridis")
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        for i, value in enumerate(values):
            plt.text(i, value + 0.02, f"{value:.4f}", ha = 'center')
        plt.show()

        # Plotting confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=set(all_targets), yticklabels=set(all_targets))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()