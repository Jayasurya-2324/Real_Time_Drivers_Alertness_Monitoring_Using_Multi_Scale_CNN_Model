#!/usr/bin/env python3
"""
train_eye_model.py - scaffold to train an eye-state classifier
Expected dataset layout (example):
  dataset/
    open/
      img1.jpg
      img2.jpg
    closed/
      imgA.jpg
      imgB.jpg

This script uses TensorFlow Keras to train a small CNN and saves models/eye_state.h5 and models/eye_state.tflite.
"""
import os, argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

def build_model(input_shape=(64,64,1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, zoom_range=0.1)
    train_gen = datagen.flow_from_directory(args.dataset, target_size=(64,64),
                                            color_mode='grayscale', batch_size=32, class_mode='binary',
                                            subset='training')
    val_gen = datagen.flow_from_directory(args.dataset, target_size=(64,64),
                                          color_mode='grayscale', batch_size=32, class_mode='binary',
                                          subset='validation')
    model = build_model(input_shape=(64,64,1))
    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen)
    os.makedirs('models', exist_ok=True)
    model.save('models/eye_state.h5')
    # TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('models/eye_state.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved models/eye_state.h5 and models/eye_state.tflite")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to dataset root with subfolders open/closed")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
