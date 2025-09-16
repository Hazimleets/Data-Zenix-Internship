# src/train.py

import tensorflow as tf
from data_loader import get_data_generators
from model import build_model
import os

# Paths
train_dir = "../data/seg_train/seg_train"
test_dir = "../data/seg_test/seg_test"
log_dir = "../logs"
model_save_path = "../saved_models/intel_model.h5"

# Parameters
img_size = (150,150)
batch_size = 32
epochs = 25

# Data generators
train_gen, test_gen = get_data_generators(train_dir, test_dir, img_size, batch_size)

# Build model
model = build_model(input_shape=(150,150,3), num_classes=6)

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Model checkpoint
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
)

# Train model
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    callbacks=[tensorboard_callback, checkpoint_callback]
)
