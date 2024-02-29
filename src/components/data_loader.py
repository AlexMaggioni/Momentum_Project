import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset


class AutoEncoderDataset(Dataset):
    def __init__(self, type_data="train"):
        super(AutoEncoderDataset, self).__init__()
        
        self.type = type_data
        
        if self.type == "train":
            # Training transformations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-70, 70)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Validation/Test transformations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
        # Load the entire dataset
        full_data = datasets.FashionMNIST(root='./data', train=(self.type != "test"), download=True)
        full_size = len(full_data)
        indices = torch.arange(full_size)
        
        if self.type == "train":
            self.indices = indices[:int(0.8 * full_size)]
        elif self.type == "val":
            self.indices = indices[int(0.8 * full_size):]
        else: 
            self.data = full_data
            return 
        
        # Apply indices for train/val
        self.data = Subset(full_data, self.indices)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        return image, image

class ClassifierDataset(Dataset):
    def __init__(self, type_data="train"):
        super(ClassifierDataset, self).__init__()

        self.type = type_data
        
        if self.type == "train":
            # Training transformations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-70, 70)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Validation/Test transformations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
        # Load the entire dataset
        full_data = datasets.FashionMNIST(root='./data', train=True, download=True)
        full_size = len(full_data)
        indices = torch.arange(full_size)
        
        if self.type == "train":
            self.indices = indices[:int(0.1 * full_size)]  # Only 10% for training
        elif self.type == "val":
            self.indices = indices[int(0.1 * full_size):]  # The rest 90% for validation
        else:  
            self.data = datasets.FashionMNIST(root='./data', train=False, download=True)
            return  
        
        # Apply indices for train/val
        self.data = Subset(full_data, self.indices)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        return image, label