import argparse

import numpy as np
import tensorflow as tf

import data
import models
import sys
import json

def get_weight():
    model_settings = models.prepare_model_settings(len(data.prepare_words_list(FLAGS.wanted_words.split(','))),
                                                   FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                                                   FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

    model = models.create_model(model_settings, FLAGS.model_architecture, FLAGS.model_size_info)
    print(len(data.prepare_words_list(FLAGS.wanted_words.split(','))),data.prepare_words_list(FLAGS.wanted_words.split(',')))
    model.load_weights(FLAGS.checkpoint).expect_partial()
    model.summary()
    model_weights = model.get_weights()
    arr=np.array(model_weights)

    #np.set_printoptions(threshold=sys.maxsize)
    #np.set_printoptions(precision=14, suppress=True)
    write_txt(arr,FLAGS.output_file)

def write_txt(data,path):
    with open(path, "w") as f:
        for i,d in enumerate(data):
            weight = {'shape':d.shape,'data':d.tolist()}
            f.write(json.dumps(weight)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be unknown words.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Checkpoint to load the weights from.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='dnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[128, 128, 128],
        help='Model dimensions - different for various models')
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file from Model\'s Weights.')

    FLAGS, _ = parser.parse_known_args()
    get_weight()
