import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carregar o dataset
features = np.load('dataset_features.npy')  # Padrão [num_samples, 2, 60, 60]
targets = np.load('dataset_targets.npy')    # Padrão [num_samples, 3]

# Separar os cortes (dois cortes por amostra)
cut1 = features[:, 0]  # Primeiro corte de cada amostra
cut2 = features[:, 1]  # Segundo corte de cada amostra

# Dividir em treino e teste (80% treino, 20% teste)
cut1_train, cut1_test, cut2_train, cut2_test, targets_train, targets_test = train_test_split(
    cut1, cut2, targets, test_size=0.2, random_state=42)

# Expandir dimensões para incluir canal de cor (necessário para CNN)
cut1_train = cut1_train[..., np.newaxis]  # Shape: (samples, 60, 60, 1)
cut2_train = cut2_train[..., np.newaxis]  # Shape: (samples, 60, 60, 1)
cut1_test = cut1_test[..., np.newaxis]
cut2_test = cut2_test[..., np.newaxis]

# Definir o modelo CNN
input1 = layers.Input(shape=(110, 110, 1))
input2 = layers.Input(shape=(110, 110, 1))

# Rede convolucional para processar as imagens
def cnn_branch(input_layer):
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return x

branch1 = cnn_branch(input1)
branch2 = cnn_branch(input2)

# Concatenar as duas saídas das redes convolucionais
combined = layers.concatenate([branch1, branch2])

# Camadas densas para regressão (previsão de dx, dy e ângulo)
x = layers.Dense(128, activation='relu')(combined)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(3)(x)  # Previsão de 3 valores: dx, dy e d_angle

# Definir o modelo completo
model = models.Model(inputs=[input1, input2], outputs=output)

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
history = model.fit([cut1_train, cut2_train], targets_train, 
                    validation_data=([cut1_test, cut2_test], targets_test),
                    epochs=20, batch_size=32)

# Avaliar o modelo
test_loss = model.evaluate([cut1_test, cut2_test], targets_test)
print(f"Test Loss: {test_loss}")

# Visualizar o histórico de treinamento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Fazer previsões no conjunto de teste
predictions = model.predict([cut1_test, cut2_test])

# Mostrar algumas previsões
num_samples_to_show = 5
for i in range(num_samples_to_show):
    print(f"Sample {i+1}")
    print(f"True values (dx, dy, angle): {targets_test[i]}")
    print(f"Predicted values (dx, dy, angle): {predictions[i]}")
    print()

    # Visualizar cortes e predições
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cut1_test[i].squeeze(), cmap='gray')
    plt.title('Cut 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cut2_test[i].squeeze(), cmap='gray')
    plt.title('Cut 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cut2_test[i].squeeze(), cmap='gray')
    plt.title('Predicted Alignment')
    plt.axis('off')

    plt.show()
