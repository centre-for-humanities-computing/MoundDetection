#!/usr/bin/python

# sys tools
import os
import pathlib
from PIL import Image

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.resnet50 import (preprocess_input,
                                                    decode_predictions,
                                                    ResNet50)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout,
                                     Conv2D,
                                     Activation, 
                                     MaxPooling2D)

# generic model object
from tensorflow.keras.models import Model, Sequential

# optimizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1,l2,l1_l2

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# for plotting
import numpy as np
import matplotlib.pyplot as plt


def plot_history(H, epochs):
    """
    Quick function to plot model history for accuracy/loss
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def load_data(train_data_dir, 
              test_data_dir, 
              batch_size=32, 
              img_height=150, 
              img_width=150, 
              split=0.2):
    """
    Load training, validation, and testing datasets
    
    Training data is augmented using augmentation layer
    """
    # Training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                  train_data_dir,
                  validation_split=0.2,
                  subset="training",
                  color_mode="rgb",
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=batch_size)
    
    # Validation
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                  train_data_dir,
                  validation_split=0.2,
                  subset="validation",
                  color_mode="rgb",
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=batch_size)

    # Test
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    test_data_dir,
                    color_mode="rgb",
                    shuffle=False,
                    image_size=(img_height, img_width))
    
    # Augment training data
    data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),])
    # Augmented dataset
    aug_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    return aug_ds, val_ds, test_ds

def cnn_model(train_ds, val_ds):
    """
    Use pretrained CNN for transfer learning
    """
    # Clear backend
    tf.keras.backend.clear_session()
    
    # Set checkpoints - train on validation loss
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints", 
                                                    mode='min', 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True)
    # Get list of callback checkpoints
    callbacks_list = [checkpoint]
    
    # load model without classifier layers
    model = ResNet50(include_top=False, 
                      pooling='avg',
                      input_shape=(150, 150, 3))
    
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, 
                   activation='relu', 
                   kernel_regularizer=l2(0.0001), 
                   bias_regularizer=l2(0.0001))(flat1)
    drop1 = Dropout(0.2)(class1)
    output = Dense(1, activation='sigmoid')(class1)

    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)
    
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy',
                           tf.keras.metrics.AUC()])
    
    # Fit model
    H = model.fit(train_ds,
                  validation_data=val_ds, 
                  batch_size=32,
                  epochs=10,
                  verbose=1,
                  callbacks=callbacks_list)
    
    # show history
    plot_history(H, 10)
    
    return model


def predict_unseen(model, test_ds):
    """
    Use trained model to predict unseen data
    """
    # Get predicted categories from model
    predicted_categories = (model.predict(test_ds) > 0.5).astype("int32")
    # Get actual categories from test_ds
    true_categories = tf.concat([y for x, y in test_ds], axis = 0).numpy()
    # sklearn classification report
    print(classification_report(true_categories, 
                                predicted_categories,
                                target_names=test_ds.class_names))
    
def main():
    # Define paths
    train_data_dir = pathlib.Path("../DATA/PNG_RGB/merged")
    test_data_dir = pathlib.Path("../DATA/PNG_RGB/test")
    # Load data
    aug_ds, val_ds, test_ds = load_data(train_data_dir, test_data_dir)
    # Train model
    best_model = cnn_model(aug_ds, val_ds)
    # predictions
    predict_unseen(best_model, test_ds)
    
    
if __name__=="__main__":
    main()