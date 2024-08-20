import numpy as np
import os

def append_to_file(output_file, data):
    # Adiciona os dados ao arquivo especificado
    with open(output_file, 'ab') as f:
        np.save(f, data)

def combine_temp_files_incrementally(temp_folder='.', output_features_file='dataset_features.npy', output_targets_file='dataset_targets.npy', num_chunks=20, chunk_size=5000):
    # Verifica se os arquivos finais já existem e os remove para iniciar uma nova criação
    if os.path.exists(output_features_file):
        os.remove(output_features_file)
    if os.path.exists(output_targets_file):
        os.remove(output_targets_file)

    for i in range(num_chunks):
        temp_features_file = os.path.join(temp_folder, f'temp_dataset_features_{i * chunk_size}.npy')
        temp_targets_file = os.path.join(temp_folder, f'temp_dataset_targets_{i * chunk_size}.npy')
        
        print(f"Processing chunk {i + 1}/{num_chunks}...")

        # Carregar dados do arquivo temporário
        features_chunk = np.load(temp_features_file)
        targets_chunk = np.load(temp_targets_file)
        
        # Adicionar ao dataset final
        append_to_file(output_features_file, features_chunk)
        append_to_file(output_targets_file, targets_chunk)

        # Liberar memória
        del features_chunk, targets_chunk

    print("Dataset final criado com sucesso!")

# Chame a função para combinar os arquivos temporários
combine_temp_files_incrementally(temp_folder='.', output_features_file='dataset_features.npy', output_targets_file='dataset_targets.npy', num_chunks=20, chunk_size=5000)
