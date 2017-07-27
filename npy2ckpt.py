#!/usr/bin/env python
import tensorflow as tf
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore debugging logs

from models.googlenet import GoogleNet


def convert(model_data_path):
    '''Convert the GoogleNet model parameters to .ckpt.'''

    # Set the data specifications for the GoogleNet model
    crop_size = 224
    channels = 3

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, crop_size, crop_size, channels),
                                name='inputs')

    # Construct the network
    net = GoogleNet({'data': input_node})

    with tf.Session() as sesh:
        # Load the converted parameters
        print('Loading the model...')
        net.load(model_data_path, sesh)

        saver = tf.train.Saver()
        save_path = saver.save(sesh, "/tmp/googlenet_model.ckpt")
        print("Model saved in file: %s" % save_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Model parameters (.npy)')
    args = parser.parse_args()

    # Convert the model parameters to .ckpt
    convert(args.model_path)


if __name__ == '__main__':
    main()
