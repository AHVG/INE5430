#import dependencies
import os
import time # Importar o módulo time

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# Define hyperparameters
seed = 42
num_epochs = 50
batch_size = 64
learning_rate = 0.0002
image_size = 28
image_channels = 1
latent_dim = 100

# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Define output directories
OUTPUT_DIR_IMAGES = 'generated_gan_images'
OUTPUT_DIR_LOSS_PLOTS = 'gan_loss_plots' # Este diretório agora conterá o gráfico de perda ao longo das épocas

os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True) # Create the directory for images
os.makedirs(OUTPUT_DIR_LOSS_PLOTS, exist_ok=True) # Create the directory for loss plots

# Create a custom dataset for MNIST
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

mnist_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

data_loader = DataLoader(dataset=mnist_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, image_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Function to save generated images
def save_generated_images(generator_model, epoch, latent_dim, output_dir, num_samples=64):
    """
    Generates sample images from the generator and saves them to a specified directory.
    """
    # Create a fixed noise vector for consistent evaluation of generator's progress
    if not hasattr(save_generated_images, 'fixed_noise'):
        save_generated_images.fixed_noise = torch.randn(num_samples, latent_dim, 1, 1)

    with torch.no_grad():
        generator_model.eval() # Set generator to evaluation mode
        fake_samples = generator_model(save_generated_images.fixed_noise).cpu()
        generator_model.train() # Set generator back to training mode

        # Create a grid of images
        fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)
        
        # Convert to numpy array and transpose for matplotlib
        np_grid = np.transpose(fake_grid.numpy(), (1, 2, 0))

        plt.figure(figsize=(8, 8))
        plt.imshow(np_grid)
        plt.title(f'Generated Images - Epoch {epoch+1}')
        plt.axis('off')
        
        # Save the image
        filename = os.path.join(output_dir, f'generated_epoch_{epoch+1:04d}.png')
        plt.savefig(filename)
        plt.close() # Close the plot to free up memory

# Function to save loss plots across epochs
def save_loss_plot_across_epochs(epoch, all_d_losses, all_g_losses, output_dir):
    """
    Plots and saves the discriminator and generator losses across all epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(all_d_losses, label='Discriminator Loss')
    plt.plot(all_g_losses, label='Generator Loss')
    plt.title(f'GAN Loss Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    
    # Always save with a fixed name as it's updated each epoch
    filename = os.path.join(output_dir, f'gan_loss_epoch_{epoch+1:04d}.png')
    plt.savefig(filename)
    plt.close() # Close the plot to free up memory


# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define loss and optimizers
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Lists to store average losses per epoch for plotting across epochs
all_d_losses = []
all_g_losses = []

# Training loop
start_training_time = time.time() # Tempo de início total do treinamento

for epoch in range(num_epochs):
    epoch_start_time = time.time() # Tempo de início da época atual

    # Lists to store losses for current epoch (for calculating average)
    current_epoch_d_batch_losses = []
    current_epoch_g_batch_losses = []

    for i, (real_images, _) in enumerate(data_loader):
        real_images = real_images
        batch_size = real_images.size(0)

        # Train discriminator with real images
        optimizer_d.zero_grad()
        label_real = torch.ones(batch_size, 1)
        output_real = discriminator(real_images).view(-1, 1)
        loss_real = criterion(output_real, label_real)
        loss_real.backward()

        # Train discriminator with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1)
        fake_images = generator(noise)
        label_fake = torch.zeros(batch_size, 1)
        output_fake = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, label_fake)
        loss_fake.backward()
        optimizer_d.step()

        # Calculate total discriminator loss for current batch and store
        loss_d = loss_real.item() + loss_fake.item()
        current_epoch_d_batch_losses.append(loss_d)

        # Train generator
        optimizer_g.zero_grad()
        output = discriminator(fake_images).view(-1, 1)
        loss_g = criterion(output, label_real)
        loss_g.backward()
        optimizer_g.step()

        # Store generator loss for current batch
        current_epoch_g_batch_losses.append(loss_g.item())

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], '
                  f'D_real: {output_real.mean():.4f}, D_fake: {output_fake.mean():.4f}, '
                  f'Loss_D: {loss_d:.4f}, Loss_G: {loss_g.item():.4f}')
    
    # Calculate average loss for the current epoch
    avg_d_loss_epoch = np.mean(current_epoch_d_batch_losses)
    avg_g_loss_epoch = np.mean(current_epoch_g_batch_losses)
    
    all_d_losses.append(avg_d_loss_epoch)
    all_g_losses.append(avg_g_loss_epoch)

    # Save generated images at the end of each epoch
    save_generated_images(generator, epoch, latent_dim, OUTPUT_DIR_IMAGES)

    # Save loss plot across epochs (updated after each epoch)
    save_loss_plot_across_epochs(epoch, all_d_losses, all_g_losses, OUTPUT_DIR_LOSS_PLOTS)

    # Time estimation
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f'Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.')
    
    # Estimate remaining time
    if epoch == 0:
        avg_epoch_duration = epoch_duration
    else:
        avg_epoch_duration = (epoch_end_time - start_training_time) / (epoch + 1)
    
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_remaining_time = remaining_epochs * avg_epoch_duration

    print(f'Average epoch duration: {avg_epoch_duration:.2f} seconds.')
    print(f'Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes ({estimated_remaining_time:.2f} seconds).\n')


# Save the trained generator model
torch.save(generator.state_dict(), 'generator.pth')
print("Training complete. Generator model saved to 'generator.pth'.")