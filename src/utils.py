import matplotlib.pyplot as plt
import numpy as np

def visualise_reconstruction(autoencoder, loader):
    """
    Visualizes the original and reconstructed images produced by an autoencoder.
    
    Parameters:
    - autoencoder: The trained autoencoder model used for generating reconstructions.
    - loader: DataLoader containing the dataset to visualize reconstructions for.
    """
    plt.figure(figsize=(15, 6))  
    
    for batch in loader:
        images, _ = batch 
        images = images.to("cuda")

        # Generate reconstructed images using the autoencoder
        reconstructed_images = autoencoder(images)

        for i in range(5):  # Displaying 5 images and their reconstructions
            # Display original images
            plt.subplot(2, 5, i+1)  
            plt.imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.title("Original")  
            plt.axis('off')  

            # Display reconstructed images
            plt.subplot(2, 5, i+6)  
            plt.imshow(reconstructed_images[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.title("Reconstruction")  
            plt.axis('off')  
        
        plt.tight_layout()  
        plt.show()
        break  
