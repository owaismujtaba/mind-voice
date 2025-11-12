import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

import pdb

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class OvertCoverRestClassifier(tf.keras.Model):
    def __init__(self, inputShape, numClasses=3):
        super(OvertCoverRestClassifier, self).__init__()

        self.model = models.Sequential([
            layers.Input(shape=inputShape),
            layers.Reshape((inputShape[0], inputShape[1], 1)),

            layers.Conv2D(16, kernel_size=(1, 10), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(inputShape[0], 1), activation='relu', padding='valid'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(1, 4)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(numClasses, activation='softmax')
        ])

    def compileModel(self, learningRate=0.001):
        self.model.compile(
        optimizer=Adam(learning_rate=learningRate),
        loss=SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    def trainWithSplit(self, X, y, validationSplit=0.2, epochs=50, batchSize=128, shuffle=True, logger=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validationSplit, stratify=y, random_state=42, shuffle=shuffle
        )

        earlyStop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batchSize,
            callbacks=[earlyStop]
        )
        y_pred_probs = self.model.predict(X_val)
        
        # If output is one-hot encoded, convert both y_val and y_pred to class labels
        if y_pred_probs.shape[-1] > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
            
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            

        # Generate reports
        report = classification_report(y_val, y_pred, digits=4, output_dict=True)
        conf_matrix = confusion_matrix(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        hist = pd.DataFrame(history.history)
        if logger:
            for epoch in range(hist.shape[0]):
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"loss: {hist['loss'][epoch]:.4f}, "
                    f"val_loss: {hist['val_loss'][epoch]:.4f}, "
                    f"accuracy: {hist['sparse_categorical_accuracy'][epoch]:.4f}, "
                    f"val_accuracy:{hist['val_sparse_categorical_accuracy'][epoch]:.4f}")

        
        return accuracy, report, conf_matrix
    def evaluate(self, testData):
        return self.model.evaluate(testData)

    def predict(self, inputData):
        return self.model.predict(inputData)

    def summary(self):
        return self.model.summary()
