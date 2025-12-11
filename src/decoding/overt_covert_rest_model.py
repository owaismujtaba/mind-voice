import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class OvertCoverRestClassifier(tf.keras.Model):
    def __init__(self, inputShape, numClasses=3,
                 F1=8, D=2, F2=16, kernelLength=64, dropoutRate=0.5):
        super(OvertCoverRestClassifier, self).__init__()

        channels = inputShape[0]
        samples = inputShape[1]

        self.model = models.Sequential([
            layers.Input(shape=inputShape),
            layers.Reshape((channels, samples, 1)),

            # EEGNet Block
            layers.Conv2D(filters=F1,
                          kernel_size=(1, kernelLength),
                          padding='same',
                          use_bias=False),
            layers.BatchNormalization(),

            layers.DepthwiseConv2D(kernel_size=(channels, 1),
                                   use_bias=False,
                                   depth_multiplier=D,
                                   depthwise_constraint=tf.keras.constraints.MaxNorm(1)),
            layers.BatchNormalization(),
            layers.Activation('elu'),
            layers.AveragePooling2D(pool_size=(1, 4)),
            layers.Dropout(dropoutRate),

            layers.SeparableConv2D(filters=F2,
                                   kernel_size=(1, 16),
                                   use_bias=False,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu'),
            layers.AveragePooling2D(pool_size=(1, 8)),
            layers.Dropout(dropoutRate),

            layers.Flatten(),
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

        if y_pred_probs.shape[-1] > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()

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
