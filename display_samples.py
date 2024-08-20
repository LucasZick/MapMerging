import numpy as np
import matplotlib.pyplot as plt
import random

def load_and_visualize_samples(features_file='dataset_features.npy', targets_file='dataset_targets.npy', num_samples=5):
    # Carregar os dados
    features = np.load(features_file)
    targets = np.load(targets_file)
    
    # Selecionar amostras aleatórias
    sample_indices = random.sample(range(len(features)), num_samples)
    
    # Configurar subplots para visualização
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i, idx in enumerate(sample_indices):
        cut1, cut2 = features[idx]
        dx, dy, d_angle = targets[idx]
        
        # Exibir primeira imagem do par
        axes[i, 0].imshow(cut1, cmap='gray')
        axes[i, 0].set_title(f'Cut 1\nSample {idx}')
        axes[i, 0].axis('off')
        
        # Exibir segunda imagem do par
        axes[i, 1].imshow(cut2, cmap='gray')
        axes[i, 1].set_title(f'Cut 2\n(dx={dx}, dy={dy}, angle={d_angle:.2f}°)')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
load_and_visualize_samples('dataset_features.npy', 'dataset_targets.npy', num_samples=5)
