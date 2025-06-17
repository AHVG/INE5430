import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Definir a arquitetura do Gerador (DEVE SER IDÊNTICA À DO TREINAMENTO) ---
# Copie e cole a classe Generator exatamente como ela foi definida no seu código de treinamento
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Certifique-se de que estes hiperparâmetros são os mesmos usados no treinamento
        latent_dim = 100 # Dimensão latente do vetor de ruído
        image_channels = 1 # 1 para imagens em escala de cinza (MNIST)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0), # Saída: 256x7x7
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # Saída: 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, image_channels, 4, 2, 1), # Saída: 1x28x28
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# --- 2. Configurações para Geração ---
# Caminho para o arquivo de pesos do Gerador
MODEL_PATH = 'generator.pth' 

# Hiperparâmetros (devem ser os mesmos usados no treinamento)
latent_dim = 100
num_samples_to_generate = 64 # Quantas imagens você quer gerar para visualizar
image_size = 28 # Tamanho da imagem gerada (MNIST é 28x28)

# Criar diretório para salvar as imagens geradas, se não existir
OUTPUT_DIR_INFERENCE = 'generated_digits_inference'
os.makedirs(OUTPUT_DIR_INFERENCE, exist_ok=True)

# --- 3. Carregar o Modelo ---
generator = Generator()

# Verifica se o arquivo do modelo existe
if not os.path.exists(MODEL_PATH):
    print(f"Erro: Arquivo do modelo '{MODEL_PATH}' não encontrado. Certifique-se de que o treinamento foi executado e salvou o modelo.")
else:
    print(f"Carregando pesos do modelo de '{MODEL_PATH}'...")
    generator.load_state_dict(torch.load(MODEL_PATH))
    generator.eval() # Coloca o modelo em modo de avaliação (desativa dropout, batchnorm para inferência)
    print("Modelo carregado com sucesso!")

    # --- 4. Gerar Dígitos ---
    print(f"Gerando {num_samples_to_generate} novos dígitos...")
    # Cria um vetor de ruído aleatório
    # O formato deve ser (num_samples, latent_dim, 1, 1) para camadas ConvTranspose2d
    noise = torch.randn(num_samples_to_generate, latent_dim, 1, 1)

    # Gera as imagens. torch.no_grad() desativa o cálculo de gradientes para otimização de memória
    with torch.no_grad():
        generated_images = generator(noise).cpu() # Move para CPU para visualização com matplotlib

    # --- 5. Visualizar e Salvar os Dígitos Gerados ---
    # As imagens geradas pelo Tanh estarão no intervalo [-1, 1].
    # Normalizamos para [0, 1] para exibição correta.
    # make_grid organiza as imagens em um grid.
    grid = torchvision.utils.make_grid(generated_images, padding=2, normalize=True, nrow=8)
    np_grid = np.transpose(grid.numpy(), (1, 2, 0)) # Transpõe para o formato (H, W, C) para matplotlib

    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.title('Dígitos Gerados pelo Modelo Treinado')
    plt.axis('off')
    
    # Salvar a imagem do grid
    output_filename = os.path.join(OUTPUT_DIR_INFERENCE, 'generated_digits_grid.png')
    plt.savefig(output_filename)
    plt.show()
    plt.close() # Fecha a figura para liberar memória

    print(f"Dígitos gerados e salvos em '{output_filename}'")