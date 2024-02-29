import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()

        # Note here that the bias has been set to "false" because biases are redundant 
        # with the shift parameter in the batch normalisation layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = self.shortcut = nn.Sequential()
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + shortcut)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(AutoEncoder, self).__init__()
        
        # Maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        encoder_channels = [in_channels, 64, 128, 256, 512]
        for in_ch, out_ch in zip(encoder_channels, encoder_channels[1:]):
            self.encoder.append(ResidualBlock(in_ch, out_ch))
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)
        
        # Decoder
        self.decoder = nn.ModuleList()
        decoder_channels = [1024, 512, 256, 128, 64]
        for in_ch, out_ch in zip(decoder_channels, decoder_channels[1:]):
            self.decoder.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0))
            self.decoder.append(ResidualBlock(out_ch, out_ch)) 
        
        # Final layer to match the original input's channel size
        self.final = nn.Conv2d(64, in_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
            x = self.maxpool(x)
        x = self.bottleneck(x)
        return x
        
    def forward(self, x):
        
        encoder = []
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            encoder.append(x)
            x = self.maxpool(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for layer in self.decoder:
            
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                encoder_shape = encoder.pop()
                if x.shape[2:] != encoder_shape.shape[2:]:
                    x = transforms.functional.resize(x, size=(encoder_shape.shape[2:]))
            
            else:
                x = layer(x)
        
        # Final layer
        x = self.tanh(self.final(x))
        return x


class Classifier(nn.Module):

    def __init__(self, autoencoder, num_classes):
        super(Classifier, self).__init__()
        self.autoencoder = autoencoder
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5) # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.autoencoder.encode(x)
        x = self.fc1(torch.flatten(x, 1))
        x = self.dropout(x)
        x = self.fc2(x) # Return logits for nn.CrossEntropyLoss
        return x