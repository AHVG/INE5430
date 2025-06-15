import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math # Importar para usar math.ceil

def plot_gan_images_grid(output_dir='generated_gan_images'):
    """
    Plots a grid of GAN generated images from specified epochs.
    The grid dimensions (rows, cols) will adjust based on the number of images.

    Args:
        output_dir (str): Directory where the generated images are stored.
                          Defaults to 'generated_gan_images'.
    """
    # Épocas das imagens que você quer selecionar
    epochs_to_display = [1, 13 , 20, 33, 40, 50] # Alterei para 1, 10, 20, 30, 40, 50 como no seu código

    loaded_images = [] # Armazena apenas as imagens que foram carregadas com sucesso
    loaded_titles = [] # Armazena os títulos das imagens carregadas
    
    for epoch in epochs_to_display:
        filename = f'generated_epoch_{epoch:04d}.png'
        filepath = os.path.join(output_dir, filename)
        
        try:
            img = mpimg.imread(filepath)
            loaded_images.append(img)
            loaded_titles.append(f'Época {epoch}')
        except FileNotFoundError:
            print(f"Aviso: Arquivo '{filepath}' não encontrado. Pulando esta época.")
        except Exception as e:
            print(f"Erro ao carregar a imagem '{filepath}': {e}. Pulando esta época.")

    # Verifica se alguma imagem foi carregada com sucesso
    if not loaded_images:
        print("Erro: Nenhuma imagem pôde ser carregada do diretório especificado.")
        return

    # --- Calcular dimensões do grid dinamicamente ---
    num_images_to_plot = len(loaded_images)
    
    # Você pode ajustar o número de colunas conforme preferir (ex: 3, 4, 5)
    # Vamos manter 3 colunas como padrão, já que você especificou 2x3 anteriormente
    cols = 3 
    rows = math.ceil(num_images_to_plot / cols) # Calcula o número de linhas necessário
    
    print(f"Gerando grid de {num_images_to_plot} imagens em {rows} linhas por {cols} colunas.")

    # Cria a figura e os subplots com as dimensões calculadas
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4)) # Ajuste o figsize conforme necessário
    
    # Achata o array de eixos para facilitar a iteração, cuidando do caso de 1 linha
    if rows == 1:
        axes = np.array([axes]) # Garante que axes é sempre um array para .flatten()
    axes = axes.flatten()
    
    for i in range(len(axes)): # Itera sobre todos os subplots criados
        ax = axes[i]
        if i < num_images_to_plot: # Se houver uma imagem para este slot
            ax.imshow(loaded_images[i])
            ax.set_title(loaded_titles[i])
            ax.axis('off') # Remove os eixos para uma visualização mais limpa
        else:
            ax.axis('off') # Desativa os eixos para subplots vazios

    plt.suptitle("Imagens Geradas da GAN em Épocas Específicas", fontsize=16) # Título geral
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para evitar sobreposição do título
    plt.show() # Mostra o grid
    
    # Opcional: Para salvar a imagem em vez de exibir
    # plt.savefig(os.path.join(output_dir, 'gan_epochs_dynamic_grid.png'))
    # plt.close() # Fecha a figura para liberar memória

# --- Exemplo de Uso ---
# Certifique-se de que o diretório 'generated_gan_images' existe
# e contém as imagens das épocas que você deseja exibir (e.g., 1, 10, 20, 30, 40, 50).

plot_gan_images_grid()