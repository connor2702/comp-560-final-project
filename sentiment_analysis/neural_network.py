import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(max_features=10000, max_len=500):
    # Load the dataset (split between training and test sets)
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    
    print(f"Training sequences: {len(x_train)}")
    print(f"Test sequences: {len(x_test)}")
    
    # Pad sequences to the same length
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    
    print("Shape of training data:", x_train.shape)
    print("Shape of test data:", x_test.shape)
    
    return x_train, y_train, x_test, y_test

def build_model(max_features=10000, max_len=500):
    # Build the model architecture
    model = keras.models.Sequential([
        layers.Embedding(input_dim=max_features, output_dim=128, input_length=max_len),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(model, x_train, y_train, batch_size=64, epochs=5):
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    return history

def evaluate_model(model, x_test, y_test):
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    return test_loss, test_acc

def plot_history(history):
    # Plot training & validation accuracy and loss values
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    max_features = 10000  # consider the top 10,000 words
    max_len = 500         # cut texts after 500 words
    
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(max_features, max_len)
    
    # Build model
    model = build_model(max_features, max_len)
    
    # Train model
    history = train_model(model, x_train, y_train, batch_size=64, epochs=5)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Plot training history
    plot_history(history)

if __name__ == '__main__':
    main()
