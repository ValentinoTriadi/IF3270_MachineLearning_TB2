import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import requests
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Dropout, SimpleRNN, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Mengatur seed untuk reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Variabel konstan
MAX_FEATURES = 10000  # Ukuran maksimum jumlah kata unik
EMBEDDING_DIM = 128   # Dimensi embedding untuk representasi kata
SEQUENCE_LENGTH = 100  # Panjang maksimum dari setiap input sequence (jumlah kata per kalimat)
BATCH_SIZE = 32
EPOCHS = 10

class TextClassifier:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        self.train_file = os.path.join(data_dir, "train.csv")
        self.valid_file = os.path.join(data_dir, "valid.csv")
        self.test_file = os.path.join(data_dir, "test.csv")
        
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.label_map = None

        self.layer_vectorize = None
        self.model = None
        self.history = None
    
    def download_data(self):
        url_data = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian"
        
        for file_name in ["train.csv", "valid.csv", "test.csv"]:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                url = f"{url_data}/{file_name}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {file_name} successfully")
                else:
                    print(f"Failed to download {file_name}. Status code: {response.status_code}")
    
    def load_data(self):
        print("Loading data...")
        
        self.download_data()
        self.train_df = pd.read_csv(self.train_file)
        self.valid_df = pd.read_csv(self.valid_file)
        self.test_df = pd.read_csv(self.test_file)
        
        # Mapping label dari string ke integer
        labels = sorted(self.train_df['label'].unique())
        self.label_map = {label: i for i, label in enumerate(labels)}

        print(f"Data loaded. Train: {len(self.train_df)}, Validation: {len(self.valid_df)}, Test: {len(self.test_df)}")
        print(f"Labels: {self.label_map}")
    
    def preprocess_data(self):
        print("Preprocessing data...")
        
        # Membuat TextVectorization layer
        self.layer_vectorize = TextVectorization(
            max_tokens=MAX_FEATURES,
            output_mode='int',
            output_sequence_length=SEQUENCE_LENGTH
        )
        
        # Melatih TextVectorization layer dengan data pelatihan
        train_text = self.train_df['text'].values
        self.layer_vectorize.adapt(train_text)

        # Menyimpan vocabulary (data kata unik ke file JSON
        vocab = self.layer_vectorize.get_vocabulary()
        with open(os.path.join(self.data_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary ({len(vocab)} tokens) to {self.data_dir}/vocab.json")
            
        # Map label ke ID
        self.train_df['label_id'] = self.train_df['label'].map(self.label_map)
        self.valid_df['label_id'] = self.valid_df['label'].map(self.label_map)
        self.test_df['label_id'] = self.test_df['label'].map(self.label_map)
        
        print("Preprocessing complete.")
    
    def create_datasets(self):
        train_ds = tf.data.Dataset.from_tensor_slices((
            self.train_df['text'].values, 
            self.train_df['label_id'].values
        ))
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        valid_ds = tf.data.Dataset.from_tensor_slices((
            self.valid_df['text'].values, 
            self.valid_df['label_id'].values
        ))
        valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((
            self.test_df['text'].values, 
            self.test_df['label_id'].values
        ))
        test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, valid_ds, test_ds
    
    def build_model(self, rnn_layers=1, rnn_units=64, bidirectional=True):
        print(f"Building model with {rnn_layers} SimpleRNN layers, {rnn_units} units, bidirectional={bidirectional}")

        model = Sequential()
        
        # Memasukkan input layer dan TextVectorization layer
        model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model.add(self.layer_vectorize)
        
        # Memasukkan embedding layer
        model.add(Embedding(
            input_dim=MAX_FEATURES + 1,  # +1 for the 0 padding
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ))
        
        # Memasukkan RNN layers
        for i in range(rnn_layers):
            return_sequences = i < rnn_layers - 1  # Return sequences for all but the last layer
            
            rnn_layer = SimpleRNN(rnn_units, return_sequences=return_sequences)
            
            if bidirectional:
                model.add(Bidirectional(rnn_layer))
            else:
                model.add(rnn_layer)
            
            # Menambahkan dropout untuk regularisasi
            model.add(Dropout(0.2))
        
        # Menambahkan output layer
        model.add(Dense(len(self.label_map), activation='softmax'))
        
        # Kompilasi model dengan loss function, optimizer, dan metrics
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        model.summary()
        return model
    
    def train_model(self, model_name="rnn_model"):
        if self.model is None:
            raise ValueError("Model not built.")
        
        train_ds, valid_ds, _ = self.create_datasets()
        
        # Callbacks untuk mencegah overfitting dan menyimpan model terbaik
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'models/{model_name}.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        os.makedirs('models', exist_ok=True)

        print(f"Training model {model_name}...")
        history = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        self.history = history
        
        return history
    
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained.")

        # Load dataset test
        _, _, test_ds = self.create_datasets()
        
        # Evaluasi model pada dataset test
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(test_ds)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        y_pred = []
        y_true = []
        
        for texts, labels in test_ds:
            predictions = self.model.predict(texts)
            pred_labels = tf.argmax(predictions, axis=1).numpy()
            y_pred.extend(pred_labels)
            y_true.extend(labels.numpy())
        
        # Menghitung F1 score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        print(f"Macro F1 Score: {f1_macro:.4f}")
        
        # Classification report
        label_names = [k for k, v in sorted(self.label_map.items(), key=lambda item: item[1])]
        print(classification_report(y_true, y_pred, target_names=label_names))
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }

def plot_loss_comparison(histories, model_names, save_path='images/loss_comparison.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot semua training loss 
    for history, name in zip(histories, model_names):
        ax1.plot(history.history['loss'], label=name)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, axis='y')
    ax1.legend()
    
    # Plot semua validation loss 
    for history, name in zip(histories, model_names):
        ax2.plot(history.history['val_loss'], label=name)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

    
def plot_history(self, model_name="rnn_model"):
    if self.history is None:
        raise ValueError("No training history available.")
        
    os.makedirs('images', exist_ok=True)
        
    # Plot accuracy
    plt.figure(figsize=(12, 4))
        
    plt.subplot(1, 2, 1)
    plt.plot(self.history.history['accuracy'])
    plt.plot(self.history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
        
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
        
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_history.png')
    # plt.show()

# Main function 
def run_experiments():
    # Inisialisasi TextClassifier
    classifier = TextClassifier()
    
    # Load and preprocess data
    classifier.load_data()
    classifier.preprocess_data()
    
    # Inisialisasi dictionary untuk menyimpan hasil eksperimen
    results = {}
    
    # Experiment 1: Pengaruh Jumlah Layer SimpleRNN
    print("\n~~~~ Experiment 1: Impact of Number of SimpleRNN Layers ~~~~")
    layer_variations = [25, 50, 100]
    histories_layers = []
    names_layers = []
    layer_results = {}
    
    for layers in layer_variations:
        model_name = f"rnn_layers_{layers}"
        print(f"\nTraining model with {layers} SimpleRNN layers...")
        
        # Build dan train model
        classifier.build_model(rnn_layers=layers, rnn_units=64, bidirectional=True)
        history = classifier.train_model(model_name=model_name)

        histories_layers.append(history)
        names_layers.append(f"{layers} layer")
        
        # Evaluasi model
        metrics = classifier.evaluate_model()
        plot_history(classifier, model_name=model_name)

        # Menyimpan hasil
        layer_results[layers] = metrics
    
    plot_loss_comparison(histories_layers, names_layers, save_path='images/layers_loss_comparison.png')
    results['rnn_layers'] = layer_results
    
    # Experiment 2: Pengaruh Jumlah SimpleRNN Cells per Layer
    print("\n~~~~ Experiment 2: Impact of Number of SimpleRNN Cells per Layer ~~~~")
    unit_variations = [32, 64, 128]
    unit_results = {}

    histories_units = []
    names_units     = []
    
    for units in unit_variations:
        model_name = f"rnn_units_{units}"
        print(f"\nTraining model with {units} SimpleRNN units per layer...")
        
        # Build dan train model
        classifier.build_model(rnn_layers=2, rnn_units=units, bidirectional=True)
        classifier.train_model(model_name=model_name)

        histories_units.append(classifier.history)
        names_units.append(f"{units} units")
        
        # Evaluasi model
        metrics = classifier.evaluate_model()
        plot_history(classifier, model_name=model_name)
        
        # Menyimpan hasil
        unit_results[units] = metrics

    plot_loss_comparison(histories_units, names_units, save_path='images/units_loss_comparison.png')
    results['rnn_units'] = unit_results
    
    # Experiment 3: Pengaruh Tipe Layer SimpleRNN Berdasarkan Arah
    print("\n~~~~ Experiment 3: Impact of SimpleRNN Layer Type by Direction ~~~~")
    direction_variations = [True, False]  # True = bidirectional, False = unidirectional
    direction_results = {}

    histories_direction = []
    names_direction    = []
    
    for bidirectional in direction_variations:
        direction_type = "bidirectional" if bidirectional else "unidirectional"
        model_name = f"rnn_{direction_type}"
        print(f"\nTraining {direction_type} SimpleRNN model...")
        
        # Build dan train model
        classifier.build_model(rnn_layers=2, rnn_units=64, bidirectional=bidirectional)
        classifier.train_model(model_name=model_name)

        histories_direction.append(classifier.history)
        names_direction.append(direction_type)
        
        # Evaluasi model
        metrics = classifier.evaluate_model()
        plot_history(classifier, model_name=model_name)
        
        # Menyimpan hasil
        direction_results[direction_type] = metrics
    
    plot_loss_comparison(histories_direction, names_direction, save_path='images/direction_loss_comparison.png')                
    results['rnn_direction'] = direction_results
    
    # Menyimpan hasil ke JSON file
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Summary Hasil
    print("\n~~~~ Experimental Results ~~~~")
    print("\nExperiment 1: Impact of Number of SimpleRNN Layers")
    for layers, metrics in results['rnn_layers'].items():
        print(f"{layers} layers: F1-Macro = {metrics['f1_macro']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    print("\nExperiment 2: Impact of Number of SimpleRNN Cells per Layer")
    for units, metrics in results['rnn_units'].items():
        print(f"{units} units: F1-Macro = {metrics['f1_macro']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    print("\nExperiment 3: Impact of SimpleRNN Layer Type by Direction")
    for direction, metrics in results['rnn_direction'].items():
        print(f"{direction}: F1-Macro = {metrics['f1_macro']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    best_f1 = 0
    best_config = ""
    
    for experiment, configs in results.items():
        for config, metrics in configs.items():
            if metrics['f1_macro'] > best_f1:
                best_f1 = metrics['f1_macro']
                best_config = f"{experiment}_{config}"
    
    print(f"\nBest model: {best_config} with F1-Macro = {best_f1:.4f}")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    run_experiments() 