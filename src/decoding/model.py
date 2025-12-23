import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from src.utils import set_all_seeds

set_all_seeds()


class OvertCoverRestClassifier(tf.keras.Model):
    def __init__(self, input_shape, num_classes=3, F1=8, D=2, 
                 F2=16, kernel_length=64, dropout_rate=0.5):
        super(OvertCoverRestClassifier, self).__init__()

        channels = input_shape[0]
        samples = input_shape[1]

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((channels, samples, 1)),

            # EEGNet Block 1: Temporal Convolution
            layers.Conv2D(filters=F1,
                          kernel_size=(1, kernel_length),
                          padding='same',
                          use_bias=False),
            layers.BatchNormalization(),

            # EEGNet Block 2: Depthwise Convolution (Spatial Filter)
            layers.DepthwiseConv2D(
                kernel_size=(channels, 1),
                use_bias=False,
                depth_multiplier=D,
                depthwise_constraint=tf.keras.constraints.MaxNorm(1)
            ),
            layers.BatchNormalization(),
            layers.Activation('elu'),
            layers.AveragePooling2D(pool_size=(1, 4)),
            layers.Dropout(dropout_rate),

            # EEGNet Block 3: Separable Convolution
            layers.SeparableConv2D(filters=F2,
                                   kernel_size=(1, 16),
                                   use_bias=False,
                                   padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu'),
            layers.AveragePooling2D(pool_size=(1, 8)),
            layers.Dropout(dropout_rate),

            layers.Flatten(),
            layers.Dense(num_classes, activation='softmax')
        ])

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy()]
        )

    def train_with_split(self, x, y, validation_split=0.2,epochs=50,
            batch_size=128, shuffle=True, logger=None
        ):
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, 
            test_size=validation_split, 
            stratify=y, 
            random_state=42, 
            shuffle=True
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )

        y_pred_probs = self.model.predict(x_val)

        if y_pred_probs.shape[-1] > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        report = classification_report(y_val, y_pred, digits=4, output_dict=True)
        conf_matrix = confusion_matrix(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        hist = pd.DataFrame(history.history)
        if logger:
            for epoch in range(hist.shape[0]):
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"loss: {hist['loss'][epoch]:.4f}, "
                    f"val_loss: {hist['val_loss'][epoch]:.4f}, "
                    f"acc: {hist['sparse_categorical_accuracy'][epoch]:.4f}, "
                    f"val_acc:{hist['val_sparse_categorical_accuracy'][epoch]:.4f}"
                )

        return acc, report, conf_matrix

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def summary(self):
        return self.model.summary()