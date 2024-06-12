import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch
import keras_tuner as kt
import pickle

# Constrói modelo com vários hiperparâmetros que serão usados para encontrar o melhor modelo
def build_model(hp):
    # Sequencial, uma camada após a outra
    model = Sequential()
    
    # Conv2D layers variando de 1 a 3
    for i in range(hp.Int('conv_layers', 1, 3)):
         # Prover input para todas as camamadas convolucionais
        input_shape = (28, 28, 1) if i == 0 else model.layers[-1].output_shape

        # Adiciona uma nova camada convolucional com filtros variando de 32 a 128 e kernels variando de 3x3 até 5x5,
        # com função de ativação Relu em cada camada
        model.add(Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
            activation='relu',
            input_shape=input_shape
        ))

        #Adiciona um max pooling de 2x2 ou 3x3
        model.add(MaxPooling2D(
            pool_size=hp.Choice(f'pool_size_{i}', values=[2, 3])
        ))
    
    # Adiciona flatten que transforma a matriz multidimensional de dados (imagem 2D) em um vetor unidimensional (1D).
    model.add(Flatten())
    
    # Camadas densas
    for i in range(hp.Int('dense_layers', 1, 3)):
        #Adiciona uma camada densa com 64 até 256 neurônios e função de ativação Relu
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=64),
            activation='relu'
        ))
        # Adiciona o dropout que adiciona ruído na rede, isto é, desativa aleatoriamente de 20% a 50% dos neurônios para
        # melhorar a adaptabilidade da rede e não viciá-la nos dados de treinamento
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Camada densa final, com 10 neurônios (´para classificar de 0 a 9) e 
    # função de ativação softmax, ideal para problemas de classificação
    model.add(Dense(10, activation='softmax'))

    # Compila o modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
    
    return model

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

# Separa o treinamento em dois: treinamento e validação
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definição da busca por melhores hiperparâmetros
tuner = kt.Hyperband(build_model,
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=3,
                    directory='my_dir',
                    project_name='intro_to_kt')

#defini
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Executar a busca de hiperparâmetros
tuner.search(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation), batch_size=200, verbose=1, callbacks=[stop_early])

# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Melhores hiperparâmetros: {best_hps.values}")

# Constrói o modelo com os melhores hiperparâmetros por 50 épocas
hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=50, batch_size=200, verbose=1)

# Identifica a época com a melhor acurácia
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Cria o modelo com os melhores hiperparâmetros novamente
model = tuner.hypermodel.build(best_hps)

# Guardar os pesos iniciais
initial_weights = [layer.get_weights() for layer in model.layers]

# Retreinar o modelo com a quantidade de épocas ideais
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=best_epoch, batch_size=200, verbose=1)

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

hyperparameters = best_hps

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
