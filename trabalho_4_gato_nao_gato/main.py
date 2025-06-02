import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras import layers, models


number_of_training_cases = 209
number_of_test_cases = 50
train_file = "train_catvnoncat.h5"
test_file = "test_catvnoncat.h5"

# Configuração para o Perceptron (Regressão Logística)
PERCEPTRON_CONFIG = {
    'name': 'Perceptron (Regressão Logística)',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 100,
    'batch_size': int(number_of_training_cases / 10),
    'verbose': 0,
    'optimizer': 'adam',
    'model_layers': [
        layers.Input(shape=(12288,)),
        layers.Dense(1, activation='sigmoid')
    ]
}

# Configuração para a Rede Neural Rasa
SHALLOW_NN_CONFIG = {
    'name': 'Rede Neural Rasa',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 100,
    'batch_size': int(number_of_training_cases / 5),
    'verbose': 0,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
    'model_layers': [
        layers.Input(shape=(12288,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
}

# Configuração para a Rede Neural Convolucional (CNN)
CNN_CONFIG = {
    'name': 'Rede Neural Convolucional (CNN)',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 50,
    'batch_size': int(number_of_training_cases / 5),
    'verbose': 0,
    'optimizer': 'adam',
    'model_layers': [
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
}


def load_dataset():
    with h5py.File(train_file, 'r') as train_dataset:
        train_set_x = np.array(train_dataset["train_set_x"][:])
        train_set_y = np.array(train_dataset["train_set_y"][:])

    with h5py.File(test_file, 'r') as test_dataset:
        test_set_x = np.array(test_dataset["test_set_x"][:])
        test_set_y = np.array(test_dataset["test_set_y"][:])

    train_set_y = train_set_y.reshape((train_set_y.shape[0], 1))
    test_set_y = test_set_y.reshape((test_set_y.shape[0], 1))
    return train_set_x, train_set_y, test_set_x, test_set_y

def preprocess_flatten(X):
    X = X / 255.0
    return X.reshape(X.shape[0], -1)

def preprocess_cnn(X):
    return X / 255.0


class BaseModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.name = config.get('name', 'Modelo Genérico de IA')

    def _build_model(self):
        """
        Builds the Keras Sequential model from the list of layers defined in config.
        The input_shape is expected to be defined within the first layer in 'model_layers'.

        Returns:
            tensorflow.keras.Model: The constructed Keras model.
        """
        model = models.Sequential()
        
        layers_to_add = self.config.get('model_layers')
        if not isinstance(layers_to_add, list) or not layers_to_add:
            raise ValueError("The 'model_layers' key must be a non-empty list of Keras Layer objects in the configuration.")
            
        # Add each layer object directly to the model
        for layer_obj in layers_to_add:
            model.add(layer_obj)
            
        return model

    def _compile_model(self):
        """
        Compiles the Keras model with the optimizer, loss function, and metrics from the configuration.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() before compiling.")
        self.model.compile(
            optimizer=self.config['optimizer'],
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Constructs, compiles, and trains the model.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
        """
        print(f"Building and training: {self.name}")
        self.model = self._build_model() # Input shape is defined within the layers now
        self.model.summary()
        self._compile_model()

        print(f"Starting training (epochs={self.config['epochs']}, batch_size={self.config['batch_size']}, verbose={self.config['verbose']})...")
        
        self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=self.config['verbose']
        )
        print("Training completed.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates the trained model and displays performance metrics and the confusion matrix.

        Args:
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"\nEvaluating: {self.name}")
        
        # Evaluate model directly with Keras for loss and metrics from compilation
        _, keras_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Keras Accuracy (from model.evaluate): {keras_accuracy:.4f}")

        # Get predictions (probabilities)
        y_pred_proba = self.model.predict(X_test)
        
        # Convert probabilities to binary class predictions (0 or 1)
        # Flatten y_test and y_pred for scikit-learn metrics if they are (N, 1)
        y_test_flat = y_test.flatten()
        y_pred_proba_flat = y_pred_proba.flatten()
        y_pred_binary = (y_pred_proba_flat > 0.5).astype(int)

        # Scikit-learn accuracy
        sk_accuracy = accuracy_score(y_test_flat, y_pred_binary)
        print(f"Scikit-learn Accuracy (thresholded): {sk_accuracy:.4f}")

        print("\nConfusion Matrix:")
        # Use y_test_flat and y_pred_binary for ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(
            y_test_flat,
            y_pred_binary,
            display_labels=["Não Gato", "Gato"], # Labels for your classes
            cmap=plt.cm.Blues
        )
        plt.title(f"Matriz de Confusão para {self.name}")

        # Cria um nome de arquivo seguro usando o nome do modelo
        file_name = f"confusion_matrix_{self.name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        
        # Opcional: Criar um diretório para salvar as imagens se ele não existir
        output_dir = "confusion_matrices"
        os.makedirs(output_dir, exist_ok=True) # Cria o diretório se não existir
        save_path = os.path.join(output_dir, file_name)

        plt.savefig(save_path) # Salva a figura antes de mostrar
        print(f"Matriz de confusão salva em: {save_path}")
        
        plt.show()

        print("\nRelatório de Classificação:")
        print(classification_report(y_test_flat, y_pred_binary, target_names=["Não Gato", "Gato"]))


class PerceptronTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__(PERCEPTRON_CONFIG)

class ShallowNNTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__(SHALLOW_NN_CONFIG)

class CNNTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__(CNN_CONFIG)


def main():
    """
    Main function to load data, preprocess,
    train, and evaluate the different models.
    """
    X_train, y_train, X_test, y_test = load_dataset()

    # Flattened data for Perceptron and Shallow Neural Network
    X_train_flat = preprocess_flatten(X_train)
    X_test_flat = preprocess_flatten(X_test)
    
    # Data for CNN (normalization only)
    X_train_cnn = preprocess_cnn(X_train)
    X_test_cnn = preprocess_cnn(X_test)

    # # --- Perceptron Training and Evaluation ---
    print("\n" + "="*40)
    perceptron_trainer = PerceptronTrainer()
    perceptron_trainer.train(X_train_flat, y_train)
    perceptron_trainer.evaluate(X_test_flat, y_test)

    # --- Shallow Neural Network Training and Evaluation ---
    print("\n" + "="*40)
    shallow_nn_trainer = ShallowNNTrainer()
    shallow_nn_trainer.train(X_train_flat, y_train)
    shallow_nn_trainer.evaluate(X_test_flat, y_test)

    # # --- Convolutional Neural Network (CNN) Training and Evaluation ---
    print("\n" + "="*40)
    cnn_trainer = CNNTrainer()
    cnn_trainer.train(X_train_cnn, y_train)
    cnn_trainer.evaluate(X_test_cnn, y_test)

    print("\nAll training and evaluations completed.")

if __name__ == "__main__":
    main()