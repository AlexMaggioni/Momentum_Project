import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset


class AutoEncoderDataset(Dataset):

    def __init__(self, train=True):
        super(AutoEncoderDataset, self).__init__()

        self.train_flag = train

        # Apply data augmentation for image reconstruction
        if self.train_flag:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-70, 70)),
                transforms.Normalize((0.5,), (0.5,))
                ])
        # We don't want to apply data augmentation for test dataset
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Apply same normalization as train dataset
                transforms.Normalize((0.5, ), (0.5,))
                ])

        self.data = datasets.FashionMNIST(root='./data', train=self.train_flag, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        transformed_image = self.transform(image)

        # We are trying to reconstruct the image, so we will return the same image as input and output
        return transformed_image, transformed_image


class ClassifierDataset(Dataset):

    def __init__(self, train=True):
        super(ClassifierDataset, self).__init__()

        self.train_flag = train

        # Apply data augmentation for image reconstruction
        if self.train_flag:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-70, 70)),
                transforms.Normalize((0.5,), (0.5,))
                ])

        # We don't want to apply data augmentation for test dataset
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Apply same normalization as train dataset
                transforms.Normalize((0.5, ), (0.5,))
                ])

        self.data = datasets.FashionMNIST(root='./data', train=self.train_flag, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        transformed_image = self.transform(image)

        # Normal FashionMNIST dataset returns image and label
        return transformed_image, label