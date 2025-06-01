import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras import layers, models


# --- Constantes para Regressão Logística ---
LOGISTIC_REGRESSION_LOSS_FUNCTION = 'binary_crossentropy'
LOGISTIC_REGRESSION_METRICS = ['accuracy']
LOGISTIC_REGRESSION_EPOCHS = 100
LOGISTIC_REGRESSION_BATCH_SIZE = 32
LOGISTIC_REGRESSION_VERBOSE_MODE = 0
LOGISTIC_REGRESSION_OPTIMIZER = 'adam'

# --- Constantes para Rede Neural Rasa ---
SHALLOW_NN_LOSS_FUNCTION = 'binary_crossentropy'
SHALLOW_NN_METRICS = ['accuracy']
SHALLOW_NN_EPOCHS = 100
SHALLOW_NN_BATCH_SIZE = 32
SHALLOW_NN_VERBOSE_MODE = 0
SHALLOW_NN_OPTIMIZER = 'adam'

# --- Constantes para Rede Convolucional (CNN) ---
CNN_LOSS_FUNCTION = 'binary_crossentropy'
CNN_METRICS = ['accuracy']
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 32
CNN_VERBOSE_MODE = 0
CNN_OPTIMIZER = 'adam'


# Função para carregar os dados
def load_dataset():
    with h5py.File('train_catvnoncat.h5', 'r') as train_dataset:
        train_set_x = np.array(train_dataset["train_set_x"][:])  # imagens
        train_set_y = np.array(train_dataset["train_set_y"][:])  # labels

    with h5py.File('test_catvnoncat.h5', 'r') as test_dataset:
        test_set_x = np.array(test_dataset["test_set_x"][:])
        test_set_y = np.array(test_dataset["test_set_y"][:])


    # reshape dos rótulos para (m, 1)
    train_set_y = train_set_y.reshape((train_set_y.shape[0], 1))
    test_set_y = test_set_y.reshape((test_set_y.shape[0], 1))
    return train_set_x, train_set_y, test_set_x, test_set_y

# Normalização e flatten se necessário
def preprocess_flatten(X):
    X = X / 255.0
    return X.reshape(X.shape[0], -1)

def preprocess_cnn(X):
    return X / 255.0

# Regressão Logística (Perceptron)
def logistic_regression_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=LOGISTIC_REGRESSION_OPTIMIZER, loss=LOGISTIC_REGRESSION_LOSS_FUNCTION, metrics=LOGISTIC_REGRESSION_METRICS)
    return model

# Rede Neural Rasa
def shallow_neural_net(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SHALLOW_NN_OPTIMIZER, loss=SHALLOW_NN_LOSS_FUNCTION, metrics=SHALLOW_NN_METRICS)
    return model

# Rede Convolucional (opcional)
def cnn_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=CNN_OPTIMIZER, loss=CNN_LOSS_FUNCTION, metrics=CNN_METRICS)
    return model

# Avaliação
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia: {accuracy:.4f}")
    y_pred = model.predict(X_test) > 0.5
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Imprime matriz de confusão bonita
    ConfusionMatrixDisplay.from_predictions(
        y_test.flatten(),  # Rótulos verdadeiros
        y_pred.flatten(),  # Rótulos preditos
        display_labels=["Não Gato", "Gato"],
        cmap=plt.cm.Blues
    )
    plt.title("Matriz de Confusão")
    plt.show()

# Execução principal
def main():
    X_train, y_train, X_test, y_test = load_dataset()

    # Regressão Logística
    print("\n--- Regressão Logística ---")
    X_train_flat = preprocess_flatten(X_train)
    X_test_flat = preprocess_flatten(X_test)
    model_log = logistic_regression_model(X_train_flat.shape[1:])
    model_log.summary()
    model_log.fit(X_train_flat, y_train, epochs=LOGISTIC_REGRESSION_EPOCHS, batch_size=LOGISTIC_REGRESSION_BATCH_SIZE, verbose=LOGISTIC_REGRESSION_VERBOSE_MODE)
    evaluate_model(model_log, X_test_flat, y_test)

    # Rede Rasa
    print("\n--- Rede Neural Rasa ---")
    model_rasa = shallow_neural_net(X_train_flat.shape[1:])
    model_rasa.summary()
    model_rasa.fit(X_train_flat, y_train, epochs=SHALLOW_NN_EPOCHS, batch_size=SHALLOW_NN_BATCH_SIZE, verbose=SHALLOW_NN_VERBOSE_MODE)
    evaluate_model(model_rasa, X_test_flat, y_test)

    # CNN (opcional)
    print("\n--- Rede Convolucional (CNN) ---")
    X_train_cnn = preprocess_cnn(X_train)
    X_test_cnn = preprocess_cnn(X_test)
    model_cnn = cnn_model()
    model_cnn.summary()
    model_cnn.fit(X_train_cnn, y_train, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, verbose=CNN_VERBOSE_MODE)
    evaluate_model(model_cnn, X_test_cnn, y_test)


if __name__ == "__main__":
    main()
