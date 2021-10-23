import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import math
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import config

from lenet import lenet

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=False,
    as_supervised=True,  # will return tuple (img, label) otherwise dict
    with_info=True,  # able to get info about dataset
)


@tf.function
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


@tf.function
def color(x):
    x = tf.image.random_brightness(x, max_delta=0.20)
    x = tf.image.random_contrast(x, lower=0.4, upper=1.6)
    return x


@tf.function
def hue_saturation(x):
    x = tf.image.random_hue(x, max_delta=0.20)
    x = tf.image.random_saturation(x, lower=0.4, upper=1.6)
    return x


@tf.function
def rotate(x, max_degrees=30):
    degrees = tf.random.uniform([], -max_degrees, max_degrees, dtype=tf.int64)
    x = tfa.image.rotate(x, tf.cast(degrees, tf.float32) * math.pi / 180, interpolation='BILINEAR')
    return x


@tf.function
def minimize_image(x, size=48):
    x = tf.image.resize_with_pad(x, size, 28)
    x = tf.image.resize_with_pad(x, 28, size)
    x = tf.image.resize(x, size=[28, 28])
    return x


@tf.function
def inverse(x):
    mul = tf.constant(-1.0, shape=[1, 28, 28, 1])
    ssum = tf.constant(1.0, shape=[1, 28, 28, 1])
    x = tf.multiply(x, mul)
    x = tf.add(x, ssum)
    x = tf.reshape(x, (28, 28, 1))
    return x


@tf.function
def stack(x):
    x_stack = tf.zeros([28, 28, 1], tf.float32)
    x_stack_bottom = tf.zeros([1, 28, 28, 1], tf.float32)
    x = tf.stack([x, x_stack], axis=0)
    x = tf.stack([x, x_stack_bottom], axis=1)
    return x


@tf.function
def augmentation_apply(val=0.5):
    prob = tf.random.uniform([], 0, 1, dtype=tf.float32)

    if prob < val:
        return True
    else:
        return False


@tf.function
def augment(image, label):
    # image = tf.image.central_crop(image, central_fraction=0.8) if augmentation_apply() else image

    image = image * (-1) if augmentation_apply() else image
    image = minimize_image(image) if augmentation_apply() else image
    image = rotate(image, max_degrees=25) if augmentation_apply() else image
    image = color(image) if augmentation_apply() else image
    image = inverse(image) if augmentation_apply() else image

    image = tf.image.resize(image, size=[28, 28])

    return image, label


# Representative dataset
def representative_dataset(dataset):
    def _data_gen():
        for data in dataset:
            yield [data]

    return _data_gen


def eval_tflite(tflite_model, dataset):
    """Evaluates tensorflow lite classification model with the given dataset."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    results = []
    labels = []

    for data in representative_dataset(dataset)():
        interpreter.set_tensor(input_idx, tf.reshape(data[0][0], (1, 28, 28, 1)))
        interpreter.invoke()
        results.append(interpreter.get_tensor(output_idx).flatten())
        labels.append(data[0][1].numpy())

    results = np.array(results)
    gt_labels = np.array(labels)
    # gt_labels = data[0][1]
    # gt_labels = np.array(list(dataset.map(lambda data: data['label'] + 1)))
    accuracy = (
            np.sum(np.argsort(results, axis=1)[:, -1:] == gt_labels.reshape(-1, 1)) /
            gt_labels.size)
    print(f'Accuracy (quantized): {accuracy * 100:.2f}%')


def visualization(ds_train):
    for x, y in ds_train:
        X = x.numpy().reshape((-1, 28, 28))
        Y = y.numpy()
        break

    plt.figure(facecolor="white", figsize=(10, 10))
    for i in range(config.BATCH_SIZE_TRAIN):
        ax = plt.subplot(8, 4, i + 1)
        ax.title.set_text(Y[i])
        plt.imshow(X[i, :])
        plt.grid(False)

    plt.show()


# Setup for train dataset
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(normalize_img, num_parallel_calls=config.AUTOTUNE)
ds_train = ds_train.map(augment, num_parallel_calls=config.AUTOTUNE)
ds_train = ds_train.batch(config.BATCH_SIZE_TRAIN)
ds_train = ds_train.prefetch(config.AUTOTUNE)

# Setup for test Dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=config.AUTOTUNE)
ds_test = ds_test.batch(config.BATCH_SIZE_TEST)
ds_test = ds_test.prefetch(config.AUTOTUNE)


def my_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    # x = layers.MaxPooling2D()(x)
    # x = layers.Conv2D(256, 3)(x)
    # x = layers.BatchNormalization()(x)
    # x = keras.activations.relu(x)

    x = layers.Flatten()(x)
    # x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


model = my_model()

# model = lenet([None, 28, 28, 1], num_classes=10)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

debugMode = False

if debugMode:
    visualization(ds_train)
else:
    model.fit(ds_train, epochs=config.EPOCH, verbose=2)
    loss, acc = model.evaluate(ds_test)
    print(f'Accuracy (float): {acc * 100:.2f}%')

    model.save("model")

    saved_model_dir = "model"
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    model_lite = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(model_lite)

    eval_tflite(model_lite, ds_test)
