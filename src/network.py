import tensorflow as tf
import numpy as np
import sys
import scipy.io
import h5py
import time
import os
from image_utilities import *
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))

res_dir = os.path.join(HERE,'../res')
train_dir = os.path.join(HERE, '../train')


tf.compat.v1.enable_eager_execution()


def get_network(input_dim, output_dim):
    model_input = tf.keras.layers.Input(shape=input_dim)
    layer = model_input

    large_size = 32
    small_size = 8


    layer = tf.keras.layers.Conv2D(small_size, 7, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    # layer = tf.keras.layers.MaxPool2D(3, strides=2) (layer)
    layer = tf.nn.relu(layer)

    end_block_0 = layer

    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)
    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = end_block_0 + layer
    layer = tf.nn.relu(layer)

    layer = tf.keras.layers.Conv2D(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)

    end_block_1 = layer

    layer = tf.keras.layers.Conv2D(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)
    layer = end_block_1 + layer
    layer = tf.keras.layers.Conv2D(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)

    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = end_block_1 + layer
    layer = tf.nn.relu(layer)
    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)

    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = end_block_0 + layer
    layer = tf.nn.relu(layer)
    layer = tf.keras.layers.Conv2DTranspose(small_size, 3, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)


    layer = tf.keras.layers.Conv2DTranspose(small_size, 7, 2, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)

    layer = tf.keras.layers.Conv2D(small_size, 1, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu(layer)

    layer = tf.keras.layers.Conv2D(1, 5, 1, padding='same', activation=None)(layer)
    layer = tf.keras.layers.Reshape(output_dim)(layer)
    layer = tf.nn.sigmoid(layer)

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

    test_image = load_image(os.path.join(res_dir, 'squirrel.png'))
    test_image = np.moveaxis(test_image, 1, 2)
    print(test_image.shape)
    print(images.shape)

    for epoch in range(EPOCHS):
        print(f'epoch {epoch}...')
        start_time = time.time()
        for batch in batches[:2]:
            train_step(images[batch[0]:batch[1],:,:,:], depths[batch[0]:batch[1],:,:])
        end_time = time.time()
        duration = end_time - start_time
        test_image_output = model(test_image).numpy()

        test_image_output = np.reshape(test_image_output, (640, 480, 1))
        test_image_output = cv2.cvtColor(test_image_output, cv2.COLOR_GRAY2RGB)
        test_image_output = np.moveaxis(test_image_output, 1, 0)
        save_image(test_image_output, os.path.join(train_dir, f'test_{epoch}.png'))
        print(f'duration: {duration}s')



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            print('training...')
            train_network()
        else:
            print('evaluating...')
            path_to_image_to_process = sys.argv[2]
            #case where we're evaluating the network
            #load most recent model and execute
            pass
    print('not enough arguments')




