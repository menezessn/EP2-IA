import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Carregar o dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Redimensionar as imagens para incluir a dimensão do canal
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalizar os valores dos pixels para o intervalo [0, 1]
X_train /= 255
X_test /= 255

# Converter os rótulos para o formato one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



# Construir a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Guardar os pesos iniciais
initial_weights = [layer.get_weights() for layer in model.layers]

# Treinar o modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Avaliar o modelo
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Guardar os pesos finais
final_weights = [layer.get_weights() for layer in model.layers]

# Prever os rótulos para os dados de teste
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Plotar a matriz de confusão
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotar o gráfico de loss do treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Salvar hiperparâmetros e pesos
import pickle

hyperparameters = {
    'initialization': {'input_shape': (28, 28, 1), 'num_classes': 10},
    'architecture': [
        {'layer_type': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'layer_type': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'layer_type': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'layer_type': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'layer_type': 'Dropout', 'rate': 0.25},
        {'layer_type': 'Flatten'},
        {'layer_type': 'Dense', 'units': 128, 'activation': 'relu'},
        {'layer_type': 'Dropout', 'rate': 0.5},
        {'layer_type': 'Dense', 'units': 10, 'activation': 'softmax'}
    ],
    'compile': {'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']},
    'fit': {'epochs': 10, 'batch_size': 200}
}

# Salvar hiperparâmetros
with open('hyperparameters.pkl', 'wb') as f:
    pickle.dump(hyperparameters, f)

# Salvar pesos iniciais
with open('initial_weights.pkl', 'wb') as f:
    pickle.dump(initial_weights, f)

# Salvar pesos finais
with open('final_weights.pkl', 'wb') as f:
    pickle.dump(final_weights, f)

# Salvar histórico do treinamento (erro por iteração)
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Salvar as saídas da rede neural para os dados de teste
with open('test_predictions.pkl', 'wb') as f:
    pickle.dump(y_pred, f)
