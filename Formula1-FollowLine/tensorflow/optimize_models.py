# Re-run after Kernel restart
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import pathlib
import argparse

def evaluate_model(model_path, model):
    

    model_size = os.path.getsize(model_path) / float(2**20)

def convert_baseline(model_path, model_name, tflite_models_dir):
    # convertering original model to tflite
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_file = tflite_models_dir/f"{model_name}_model.tflite"
    tflite_model_file.write_bytes(tflite_model) # save model


def dynamic_range_quantization(model_path, model_name, tflite_models_dir):
    # Post-training dynamic range quantization
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/f"{model_name}_model_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model) # save model

def load_data():
    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None,
                        help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", type=int, default=0, help="Data Augmentations: 0=No / 1=Normal / 2=Normal+Weather changes")
    parser.add_argument("--img_shape", type=str, default=(200, 66, 3), help="Image shape")
    parser.add_argument('--model_path', type=str, default='trained_models/pilotnet.h5', help="Path to directory containing pre-trained models")
    parser.add_argument('--model_name', default='pilotnet', help="Name of model" )
    # parser.add_argument('--res_path', default='Result_Model_3.csv', help="Path(+filename) to store the results" )
    parser.add_argument("--tech", action='append', help="Techniques to apply for model compression. Options are: \n"+
                               "'dynamic_quan', 'float16_quan', 'full_int_quan', and 'all' .")
    
    args = parser.parse_args()
    return args

# def tensorrt():
    # model = ResNet50()
    # model.save('resnet50_saved_model') 
    # model_path = 'resnet50_saved_model'
    # print('Converting to TF-TRT FP32...')
    # conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
    #                                                             max_workspace_size_bytes=8000000000)

    # converter = trt.TrtGraphConverterV2(input_saved_model_dir= model_path,
    #                                     conversion_params=conversion_params)
    # converter.convert()
    # converter.save(output_saved_model_dir='pilotnet_TFTRT_FP32')
    # print('Done Converting to TF-TRT FP32')


if __name__ == '__main__':

    args = parse_args()

    # model_name = "pilotnet"
    # saved_model_dir = 'trained_models/pilotnet.h5'

    # directory to save optimized models
    tflite_models_dir = pathlib.Path("tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    convert_baseline(args.model_path, args.model_name, tflite_models_dir)
    
    dynamic_range_quantization(args.model_path, args.model_name, tflite_models_dir)
    