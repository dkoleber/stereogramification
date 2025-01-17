import tensorflow as tf
from tensorflow.python.keras.saving import load_model
import numpy as np
import sys
import scipy.io
import h5py
import time
import os
from image_utilities import *
import cv2

from stereogram import get_stereogram

HERE = os.path.dirname(os.path.abspath(__file__))

res_dir = os.path.join(HERE,'../res')
train_dir = os.path.join(HERE, '../train')
models_dir = os.path.join(HERE, '../models')

tf.compat.v1.enable_eager_execution()


def get_network(input_dim, output_dim):
    model_input = tf.keras.layers.Input(shape=input_dim)
    layer = model_input

    large_size = 32
    small_size = 8


    layer = tf.keras.layers.Conv2D(small_size, 7, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    # layer = tf.keras.layers.MaxPool2D(3, strides=2) (layer)
    layer = tf.keras.layers.ReLU()(layer)

    end_block_0 = layer

    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Add()([end_block_0, layer])
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    end_block_1 = layer

    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Add()([end_block_1, layer])
    layer = tf.keras.layers.Conv2D(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Add()([end_block_1, layer])
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Add()([end_block_0, layer])
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)


    layer = tf.keras.layers.Conv2DTranspose(small_size, 7, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(small_size, 1, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(1, 5, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.Reshape(output_dim)(layer)
    layer = tf.keras.layers.ReLU()(layer)

    return tf.keras.Model(inputs=model_input, outputs=layer)

def get_nyu_dataset():
    print('loading dataset...')
    path_to_data = os.path.join(res_dir, 'nyu_depth_v2_labeled.mat')
    # loaded_mat_data = scipy.io.loadmat(path_to_data)
    with h5py.File(path_to_data, 'r') as fl:
        images = fl['images'].value
        depths = fl['depths'].value

    # images = images[:10,:,:,:]
    # depths = depths[:10,:,:]

    images = np.moveaxis(images, 1, -1) # move channel to last dimension

    images = np.true_divide(images, (255./2.))
    images = np.add(images, -1.)
    images = images.astype(np.float32)

    depths = depths.astype(np.float32)
    max_depth = np.amax(depths) / 2.
    depths = np.true_divide(depths, max_depth)
    depths = np.add(depths, -1.)

    return images, depths

def get_batches(data_length, batch_size):
    result = [(x*batch_size, (x+1)*batch_size) for x in range(int(data_length/batch_size))]
    remainder = data_length % batch_size
    remainder_start = int(data_length/batch_size) * batch_size
    result.append((remainder_start, remainder_start + remainder))
    return result

def load_most_recent_model(dir):
    models = os.listdir(dir) #load all models in dir
    models.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) #sort models by a number attached to their name
    model = load_model(os.path.join(dir, models[-1]))
    return model


def save_model(model, dir, model_name):
    model.save(os.path.join(dir, model_name + '.h5'))


def train_network():
    EPOCHS = 128
    BATCH_SIZE = 8

    images, depths = get_nyu_dataset()
    network_input_dim = images.shape[1:]
    network_output_dim = depths.shape[1:]

    model = get_network(network_input_dim, network_output_dim)
    loss_function = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam()

    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        variables = model.trainable_variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad,variables))
        return loss

    batches = get_batches(images.shape[0], BATCH_SIZE)
    #TODO: shuffle data

    test_image = load_image(os.path.join(res_dir, 'room_1.jpg'))
    test_image = np.moveaxis(test_image, 1, 2)
    print(test_image.shape)
    print(images.shape)

    for epoch in range(EPOCHS):
        print(f'epoch {epoch}...')
        start_time = time.time()
        for batch in batches:
            train_step(images[batch[0]:batch[1],:,:,:], depths[batch[0]:batch[1],:,:])
        end_time = time.time()
        duration = end_time - start_time
        test_image_output = model(test_image).numpy()

        test_image_output = np.reshape(test_image_output, (640, 480, 1))
        test_image_output = cv2.cvtColor(test_image_output, cv2.COLOR_GRAY2RGB)
        test_image_output = np.moveaxis(test_image_output, 1, 0)
        save_image(test_image_output, os.path.join(train_dir, f'test_{epoch}.png'))
        print(f'duration: {duration}s')

    save_model(model, models_dir, 'segmentation' + str(time.time()))

def make_stereogram(image_path, output_path):
    model = load_most_recent_model(models_dir)
    image = load_image(image_path)
    model_output = model(image).numpy()

    _, rows, cols = model_output.shape

    model_output = np.reshape(model_output, (rows, cols))

    padding_height = int(max(0, cols - rows) / 2)
    padding_width = int(max(0, rows - cols) / 2)
    model_output = cv2.copyMakeBorder(model_output, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rotationMatrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    model_output = cv2.warpAffine(model_output, rotationMatrix, (cols, rows))

    stereogram, circle_1, cicle_2 = get_stereogram(model_output)

    stereogram = cv2.circle(stereogram, circle_1, 10, (0, 0, 0), -1)
    stereogram = cv2.circle(stereogram, cicle_2, 10, (0, 0, 0), -1)


    rotationMatrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    stereogram = cv2.warpAffine(stereogram, rotationMatrix, (cols, rows))
    print('---')
    stereogram = stereogram.astype(np.float32)
    print(stereogram.shape)
    stereogram = cv2.cvtColor(stereogram, cv2.COLOR_GRAY2RGB)

    save_image(stereogram, output_path)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            print('training...')
            train_network()
        else:
            print('evaluating...')
            input_path = sys.argv[1]
            output_path = sys.argv[2]
            make_stereogram(input_path, output_path)
    else:
        print('not enough arguments')




