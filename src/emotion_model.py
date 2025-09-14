import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, 
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class EmotionCNN:
    """
    Convolutional Neural Network for facial emotion recognition.
    Designed to work with FER2013 dataset (48x48 grayscale images, 7 emotion classes).
    """
    
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build CNN architecture for emotion recognition."""
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(512),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(256),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer, loss, and metrics."""
        if self.model is None:
            self.build_model()
            
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_callbacks(self):
        """Get training callbacks for better convergence."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the emotion recognition model."""
        if self.model is None:
            self.compile_model()
            
        callbacks = self.get_callbacks()
        
        print("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test set."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, X):
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        predictions = self.model.predict(X)
        return predictions
    
    def predict_emotion(self, image, emotion_labels):
        """Predict emotion for a single image."""
        prediction = self.predict(image)
        emotion_idx = prediction.argmax()
        confidence = prediction[0][emotion_idx]
        emotion = emotion_labels[emotion_idx]
        
        return emotion, confidence
    
    def save_model(self, filepath):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save!")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from file."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()

if __name__ == "__main__":
    # Example usage
    cnn = EmotionCNN()
    cnn.build_model()
    cnn.compile_model()
    print("Model architecture:")
    cnn.summary()