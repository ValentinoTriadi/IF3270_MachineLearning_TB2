import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

class KerasModel:
    def __init__(self, num_class, input_shape, random_seed = 42):
        self.input_shape = input_shape
        self.num_classes = num_class
        self.random_seed = random_seed
        self.model = None
        self.history = None
        self._set_seeds()

    def _set_seeds(self):
        "Set seeds for better random"
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
    
    def preprocess_data(self, x_train_full, y_train_full, x_test_orig, y_test_orig, val_split_ratio=0.2):
        """
        Normalizes and splits the data into training, validation, and test sets.
        """
        print("Normalizing data")
        x_train_full_norm = x_train_full.astype("float32") / 255.0
        self.x_test_processed = x_test_orig.astype("float32") / 255.0
        self.y_test_processed = y_test_orig 

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train_full_norm, y_train_full,
            test_size=val_split_ratio,
            random_state=self.random_seed,
            stratify=y_train_full
        )
        print(f"x_train shape: {self.x_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"x_val shape: {self.x_val.shape}")
        print(f"y_val shape: {self.y_val.shape}")

        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test_processed, self.y_test_processed
    
    def define_model(self, model_name, conv_blocks_config, global_pooling_type, dense_layers_config):
        """
        Defines the CNN model architecture dynamically.
        """
        print(f"\nDefining Model: {model_name}")
        model_layers = [layers.Input(shape=self.input_shape, name="input_layer")]

        # Convolutional Blocks
        for i, block_config in enumerate(conv_blocks_config):
            for j in range(block_config['conv_layers_in_block']):
                model_layers.append(layers.Conv2D(
                    filters=block_config['filters'],
                    kernel_size=block_config['kernel_size'],
                    activation='relu',
                    padding='same',
                    name=f"conv_block{i+1}_layer{j+1}"
                ))

            if block_config['pooling_type'] == 'max':
                model_layers.append(layers.MaxPooling2D(pool_size=block_config.get('pooling_size', (2,2)), name=f"maxpool_block{i+1}"))
            elif block_config['pooling_type'] == 'avg':
                model_layers.append(layers.AveragePooling2D(pool_size=block_config.get('pooling_size', (2,2)), name=f"avgpool_block{i+1}"))
            # 'none' means no pooling layer for this block

            if block_config.get('dropout_after_pool', 0) > 0:
                model_layers.append(layers.Dropout(block_config['dropout_after_pool'], name=f"dropout_conv_block{i+1}"))

        # Global Pooling or Flatten
        if global_pooling_type == 'flatten':
            model_layers.append(layers.Flatten(name="flatten_layer"))
        elif global_pooling_type == 'global_avg':
            model_layers.append(layers.GlobalAveragePooling2D(name="global_avg_pool_layer"))
        elif global_pooling_type == 'global_max':
            model_layers.append(layers.GlobalMaxPooling2D(name="global_max_pool_layer"))
        else:
            raise ValueError(f"Unsupported global_pooling_type: {global_pooling_type}")

        # Dense Layers
        for i, dense_config in enumerate(dense_layers_config):
            model_layers.append(layers.Dense(dense_config['units'], activation='relu', name=f"dense_layer{i+1}"))
            if dense_config.get('dropout', 0) > 0:
                model_layers.append(layers.Dropout(dense_config['dropout'], name=f"dropout_dense_layer{i+1}"))

        model_layers.append(layers.Dense(self.num_classes, activation='softmax', name="output_layer"))

        self.model = keras.Sequential(model_layers, name=model_name)
        # self.model.summary()

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        if self.model is None:
            raise ValueError("Model has not been defined.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, epochs, batch_size, verbose=1):
        if self.model is None:
            raise ValueError("Model has not been compiled.")
        if self.x_train is None:
            raise ValueError("Data not preprocessed.")

        print(f"Training {self.model.name} for {epochs} epochs")
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=verbose
        )
        return self.history

    def evaluate_model(self, verbose=0):
        if self.model is None or self.history is None:
            raise ValueError("Model not trained.")

        loss, accuracy = self.model.evaluate(self.x_test_processed, self.y_test_processed, verbose=verbose)
        
        y_pred_proba = self.model.predict(self.x_test_processed, verbose=verbose)
        y_pred = np.argmax(y_pred_proba, axis=1)
        f1 = f1_score(self.y_test_processed, y_pred, average="macro")
        
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test Macro F1-Score: {f1:.4f}")
        return loss, accuracy, f1

    def save_model_weights(self, filepath):
        if self.model is None:
            raise ValueError("No model to save weights from.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def plot_training_history(self, experiment_name, model_name, base_save_path="experiment_results"):
        if self.history is None:
            print("No training history to plot.")
            return

        plot_dir = os.path.join(base_save_path, experiment_name, model_name)
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy: {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss: {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f"Training History for {model_name} ({experiment_name})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_path = os.path.join(plot_dir, "training_history.png")
        plt.savefig(plot_path)
        print(f"  Training history plot saved to {plot_path}")
        plt.close()

