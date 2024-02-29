import matplotlib.pyplot as plt
import numpy as np

def visualise_reconstruction(autoencoder, loader, save_path=None):
    """
    Visualizes the original, noisy, and reconstructed images produced by an autoencoder.

    Parameters:
    - autoencoder: The trained autoencoder model used for generating reconstructions.
    - loader: DataLoader containing the dataset to visualize reconstructions for.
    """
    plt.figure(figsize=(15, 9))  # Adjusted figure size to accommodate three rows

    for batch in loader:
        noisy, images = batch
        noisy = noisy.to("cuda")
        images = images.to("cuda")

        # Generate reconstructed images using the autoencoder
        reconstructed_images = autoencoder(images)
        reconstructed_images = reconstructed_images.to("cuda")

        for i in range(5): 

            # Display original images
            plt.subplot(3, 5, i + 1)
            plt.imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.title("Original")
            plt.axis('off')

            # Display reconstructed images
            plt.subplot(3, 5, i + 6)
            plt.imshow(reconstructed_images[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.title("Reconstruction")
            plt.axis('off')

            # Display noisy input images
            plt.subplot(3, 5, i + 11)
            plt.imshow(noisy[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.title("Noisy Input")
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

        break
